# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/bench/bench_mlp.py
"""Benchmark the int8 SmoothQuant MoE MLP (two moe_gemm_int8_smoothquant calls).

Each batch size is measured with `do_bench_aiter_triton` (a rotating-buffer
CUDA graph replay) and reported as three rows, keyed by the `layer` column:

  moe1   the gathered up projection alone   (N = dim2 / TP, K = dim1, + activation)
  moe2   the scattered down projection alone (N = dim1, K = dim2 / TP / 2)
  total  both back to back, i.e. the whole MLP

`total` is not `moe1 + moe2`: an isolated projection replays the same kernel
over and over, so its weights stay cache-resident in a way the real layer does
not. Compare like with like.

Weights are quantized to int8 once (per-output-channel scale); activations
are SmoothQuant-quantized to int8 fresh before each GEMM (moe1's output is
fp32, requantized before moe2 -- that requant is included in moe2's
measurement, same as the reference this benchmark is modeled on).
--preshuffled enables the MFMA-friendly weight tile layout.

Everything except the GEMMs -- gating, routing, and the activation
quantization that feeds each layer -- is built once outside the timed
region, so the numbers cover the two projections and nothing else.
"""

import csv
import functools
import inspect
from itertools import chain
from pathlib import Path

import torch

from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.moe.moe_op_gemm_int8_smoothquant import (
    moe_gemm_int8_smoothquant,
    preshuffle_weights,
)
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.quant_moe import (
    quantize_weights_int8,
    smoothquant_quantize,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
)

LAYERS = ("moe1", "moe2", "total")


def compute_roofline(
    *args, bench_fn, intensity_proxy_name, intensity_proxy_values, out_path, **kwargs
):
    if not isinstance(intensity_proxy_name, str):
        raise TypeError(
            "intensity_proxy must be a string naming a parameter in target_fn"
        )
    sig = inspect.signature(bench_fn)
    params = list(sig.parameters.values())
    if intensity_proxy_name not in sig.parameters:
        raise ValueError(
            f"Parameter '{intensity_proxy_name}' not found in {bench_fn.__name__} signature"
        )
    pos_index = [p.name for p in params].index(intensity_proxy_name)

    def inject_proxy_and_call(val, args, kwargs):
        args_list = list(args)
        args_list.insert(pos_index, val)
        return bench_fn(*args_list, **kwargs)

    perfs = []
    print("=========================================")
    print(f"{out_path}...")
    print("=========================================")

    for val in intensity_proxy_values:
        perf = inject_proxy_and_call(val, args, kwargs)
        perfs.append((val, perf))

        groups = " | ".join(
            f"{name} {lp['latency_ms'] * 1e3:.2f}us "
            f"{lp['flops'] / lp['latency_ms'] * 1e-9:#.4g} TF/s "
            f"{lp['bytes'] / lp['latency_ms'] * 1e-9:#.4g} TB/s"
            for name, lp in perf["layers"].items()
        )
        print(
            f"{intensity_proxy_name}: {val:5d} | {groups} | "
            f"block_m={perf['block_m']} routed_experts={perf['routed_experts']}"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        intensity_proxy_name,
        "layer",
        "latency_us",
        "tflops",
        "tbps",
        "flops",
        "bytes",
        "block_m",
        "routed_experts",
    ]

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for val, perf in perfs:
            for name, lp in perf["layers"].items():
                w.writerow(
                    {
                        intensity_proxy_name: val,
                        "layer": name,
                        "latency_us": lp["latency_ms"] * 1e3,
                        "tflops": lp["flops"] / lp["latency_ms"] * 1e-9,
                        "tbps": lp["bytes"] / lp["latency_ms"] * 1e-9,
                        "flops": lp["flops"],
                        "bytes": lp["bytes"],
                        "block_m": perf["block_m"],
                        "routed_experts": perf["routed_experts"],
                    }
                )


def _generate_mlp_pool_entry(
    dim1, dim2, n_expts_tot, n_expts_act, preshuffled, TP, batch
):
    """Build one full weight-init + routing + activation set.

    Pure setup, not measured. Each call is one rotating-buffer pool entry:
    independent random weights and routing, so replaying through the pool
    during timing defeats cache residency the same way a repeat-and-average
    loop would, without the cost of a fresh CUDA graph capture each time.
    """
    dev = f"cuda:{torch.cuda.current_device()}"

    wg = torch.randn((dim1, n_expts_tot), device=dev, dtype=torch.bfloat16)
    w1 = torch.randn((n_expts_tot, dim1, dim2 // TP), device=dev, dtype=torch.bfloat16)
    w2 = torch.randn(
        (n_expts_tot, dim2 // TP // 2, dim1), device=dev, dtype=torch.bfloat16
    )
    bg = torch.randn((n_expts_tot,), device=dev, dtype=torch.bfloat16)

    fc1_smooth_scale = torch.randn((dim1,), device=dev, dtype=torch.float32).abs() + 0.1
    fc2_smooth_scale = (
        torch.randn((dim2 // TP // 2,), device=dev, dtype=torch.float32).abs() + 0.1
    )

    w1_int8, w1_scale = quantize_weights_int8(w1)
    w2_int8, w2_scale = quantize_weights_int8(w2)
    w1_int8 = w1_int8.transpose(1, 2).contiguous().transpose(1, 2)
    w2_int8 = w2_int8.transpose(1, 2).contiguous().transpose(1, 2)
    if preshuffled:
        w1_int8 = preshuffle_weights(w1_int8)
        w2_int8 = preshuffle_weights(w2_int8)

    x = torch.randn((batch, dim1), dtype=torch.bfloat16, device=dev)
    logits = gemm_a16w16(x, wg.T, bg)
    rdata, gather_indx, scatter_indx = routing(logits, n_expts_act)

    x1, x1_scale = smoothquant_quantize(x, fc1_smooth_scale)
    y1 = moe_gemm_int8_smoothquant(
        x1,
        w1_int8,
        x1_scale,
        w1_scale,
        None,
        rdata,
        gather_indx=gather_indx,
        scatter_indx=None,
        preshuffled=preshuffled,
        out_dtype=torch.float32,
        apply_activation=True,
        limit=None,
        swiglu_add_residual=False,
    )
    y1_bytes = y1.numel() * y1.element_size()

    x2, x2_scale = smoothquant_quantize(y1, fc2_smooth_scale)
    y2 = moe_gemm_int8_smoothquant(
        x2,
        w2_int8,
        x2_scale,
        w2_scale,
        None,
        rdata,
        gather_indx=None,
        scatter_indx=scatter_indx,
        preshuffled=preshuffled,
        out_dtype=torch.bfloat16,
        apply_activation=False,
        swiglu_add_residual=False,
    )

    # -- analytic FLOPs / bytes: 2*M*N*K per GEMM, and activations +
    # routed-expert weights + matmul output for traffic. The moe2 scatter
    # reduction is not counted; its runtime is inside
    # moe_gemm_int8_smoothquant and so is inside the measurement.
    n_tokens = gather_indx.shape[0]  # routed rows == batch * n_expts_act
    routed = int((rdata.expt_data.hist > 0).sum())  # experts that got >= 1 token

    def w_bytes(w):
        return (w.numel() * w.element_size() // n_expts_tot) * routed

    def scale_bytes(scale):
        return (scale.numel() * scale.element_size() // n_expts_tot) * routed

    moe1_flops = 2 * n_tokens * (dim2 // TP) * dim1  # N = dim2 // TP, K = dim1
    moe1_bytes = (
        x1.numel() * x1.element_size()
        + w_bytes(w1_int8)
        + scale_bytes(w1_scale)
        + y1_bytes
    )
    moe2_flops = 2 * n_tokens * dim1 * (dim2 // TP // 2)  # N = dim1, K = dim2/TP/2
    # y2 is the scatter-compressed [batch, dim1] result; the GEMM writes the
    # uncompressed [n_tokens, dim1] rows the reduction then combines.
    moe2_bytes = (
        x2.numel() * x2.element_size()
        + w_bytes(w2_int8)
        + scale_bytes(w2_scale)
        + n_tokens * dim1 * y2.element_size()
    )

    meta = {
        "moe1_flops": moe1_flops,
        "moe1_bytes": moe1_bytes,
        "moe2_flops": moe2_flops,
        "moe2_bytes": moe2_bytes,
        "routed": routed,
        "block_m": rdata.block_m,
    }
    return (
        x1,
        w1_int8,
        x1_scale,
        w1_scale,
        x2,
        w2_int8,
        x2_scale,
        w2_scale,
        rdata,
        gather_indx,
        scatter_indx,
        meta,
    )


_MLP_POOL_FIELDS = [
    "x1",
    "w1",
    "x1_scale",
    "w1_scale",
    "x2",
    "w2",
    "x2_scale",
    "w2_scale",
    "rdata",
    "gather_indx",
    "scatter_indx",
    "meta",
]


def _make_moe_layer_fns(preshuffled):
    """Parameterized moe1/moe2/total kernel callables for do_bench_aiter_triton.

    Argument names match a rotating-buffer pool entry's keys (``**_``
    absorbs the ones a given layer does not need).
    """

    def layer1(x1, w1, x1_scale, w1_scale, rdata, gather_indx, **_):
        moe_gemm_int8_smoothquant(
            x1,
            w1,
            x1_scale,
            w1_scale,
            None,
            rdata,
            gather_indx=gather_indx,
            scatter_indx=None,
            preshuffled=preshuffled,
            out_dtype=torch.float32,
            apply_activation=True,
            limit=None,
            swiglu_add_residual=False,
        )

    def layer2(x2, w2, x2_scale, w2_scale, rdata, scatter_indx, **_):
        moe_gemm_int8_smoothquant(
            x2,
            w2,
            x2_scale,
            w2_scale,
            None,
            rdata,
            gather_indx=None,
            scatter_indx=scatter_indx,
            preshuffled=preshuffled,
            out_dtype=torch.bfloat16,
            apply_activation=False,
            swiglu_add_residual=False,
        )

    def both(**kwargs):
        layer1(**kwargs)
        layer2(**kwargs)

    return layer1, layer2, both


def bench_mlp(
    batch,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    preshuffled,
    TP,
    layers=LAYERS,
    num_weight_inits=1,
    kwargs_rot_bufs=None,
    kwargs_do_bench_graph=None,
):
    kwargs_rot_bufs = kwargs_rot_bufs or {}
    kwargs_do_bench_graph = kwargs_do_bench_graph or {}
    assert dim2 % TP == 0, f"{dim2=}, {TP=}, dim2 must be divisible by TP"
    assert layers, "at least one layer must be selected"
    assert set(layers) <= set(LAYERS), f"unknown layer(s) in {layers=}"

    # -- rotating-buffer pool: each entry is one independent weight init +
    # routing + activation set, replacing a repeat-and-average loop with one
    # capture that replays through all of them.
    pool_gen = functools.partial(
        _generate_mlp_pool_entry,
        dim1,
        dim2,
        n_expts_tot,
        n_expts_act,
        preshuffled,
        TP,
        batch,
    )
    rot_bufs = generate_rotating_buffers_pool(
        [pool_gen],
        [_MLP_POOL_FIELDS],
        target_mb=kwargs_rot_bufs.get("target_mb", 256),
        min_pool_len=num_weight_inits,
        print_info=kwargs_rot_bufs.get("print_info", False),
    )

    metas = [entry["meta"] for entry in rot_bufs]
    num_entries = len(metas)

    def avg(key):
        return sum(m[key] for m in metas) / num_entries

    moe1_flops = avg("moe1_flops")
    moe1_bytes = avg("moe1_bytes")
    moe2_flops = avg("moe2_flops")
    moe2_bytes = avg("moe2_bytes")
    routed_experts_used = avg("routed")
    # block_m depends only on batch/topk/E, not on the weight draw
    block_m = metas[0]["block_m"]

    layer1_fn, layer2_fn, both_fn = _make_moe_layer_fns(preshuffled)

    to_bench = {
        "moe1": (layer1_fn, moe1_flops, moe1_bytes),
        "moe2": (layer2_fn, moe2_flops, moe2_bytes),
        "total": (both_fn, moe1_flops + moe2_flops, moe1_bytes + moe2_bytes),
    }
    measured = {
        name: {
            "latency_ms": do_bench_aiter_triton(f, rot_bufs, **kwargs_do_bench_graph),
            "flops": flops,
            "bytes": byts,
        }
        for name, (f, flops, byts) in to_bench.items()
        if name in layers
    }

    return {
        "layers": measured,
        "block_m": block_m,
        "routed_experts": routed_experts_used,
    }


def roofline_mlp(
    batch_sizes,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    preshuffled,
    TP,
    layers=LAYERS,
    num_weight_inits=1,
    name="",
    kwargs_rot_bufs=None,
    kwargs_do_bench_graph=None,
):
    out_dir = Path("logs") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = (
        f"int8x-int8w-TP{TP}-dim1={dim1}-dim2={dim2}-E={n_expts_tot}-topk={n_expts_act}"
    )
    if preshuffled:
        stem += "-preshuffled"
    if tuple(layers) != LAYERS:
        stem += "-layers=" + "+".join(layers)
    out_csv = out_dir / f"{stem}.csv"

    compute_roofline(
        dim1,
        dim2,
        n_expts_tot,
        n_expts_act,
        preshuffled,
        TP,
        layers,
        num_weight_inits,
        bench_fn=bench_mlp,
        intensity_proxy_name="batch",
        intensity_proxy_values=batch_sizes,
        out_path=out_csv,
        kwargs_rot_bufs=kwargs_rot_bufs,
        kwargs_do_bench_graph=kwargs_do_bench_graph,
    )


def parse_args():
    parser = get_parser(kernel_name="MoE Int8 SmoothQuant GEMM")
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        metavar=("DIM"),
        help="Input feature dimensions of MoE layers. Must be two integers.",
    )
    parser.add_argument(
        "--experts",
        type=int,
        nargs="+",
        metavar=("DIM"),
        help="Number of total and active experts in [total experts, active experts] order.",
    )
    parser.add_argument(
        "--preshuffled",
        action="store_true",
        help="Preshuffle weights.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        choices=LAYERS,
        default=list(LAYERS),
        help="Which layers to measure: moe1 (up projection), moe2 (down "
        "projection), total (both back to back). Default: all three.",
    )
    parser.add_argument(
        "--num-weight-inits",
        type=int,
        default=1,
        help="Minimum number of different weight initializations in the rotating "
        "buffer pool for more stable results (default: 1). The pool can grow "
        "larger than this to reach the cache-defeating size target; it never "
        "shrinks below it. Use higher values (e.g., 10) for more stable "
        "benchmarks.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dim1, dim2 = args.shape
    total_experts, active_experts = args.experts
    batch_ranges_moe = [
        (1, 2, 1),
        (2, 5, 2),
        (8, 18, 8),
        (32, 65, 32),
        (128, 257, 128),
        (1024, 1200, 200),
        (4096, 8200, 4096),
    ]
    batch_sizes_moe = list(chain(*[range(*r) for r in batch_ranges_moe]))

    roofline_mlp(
        batch_sizes_moe,
        dim1,
        dim2,
        total_experts,
        active_experts,
        args.preshuffled,
        TP=1,
        layers=tuple(n for n in LAYERS if n in args.layers),
        num_weight_inits=args.num_weight_inits,
        name="int8-smoothquant",
        kwargs_rot_bufs=get_rot_bufs_kwargs(args),
        kwargs_do_bench_graph=get_do_bench_graph_kwargs(args),
    )
