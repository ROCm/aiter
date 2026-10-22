# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/bench/bench_mlp.py
"""Benchmark the bf16/fp16 x mxfp4 MoE MLP (two moe_gemm_a16w4 calls).

Each batch size is measured with `do_bench_aiter_triton` (a rotating-buffer
CUDA graph replay) and reported as three rows, keyed by the `layer` column:

  moe1   the gathered up projection alone   (N = dim2 / TP, K = dim1, + swiglu)
  moe2   the gathered down projection alone (N = dim1, K = dim2 / TP / 2)
  total  both back to back, i.e. the whole MLP

`total` is not `moe1 + moe2`: an isolated projection replays the same kernel
over and over, so its weights stay cache-resident in a way the real layer does
not. Compare like with like.

Activations stay bf16/fp16 throughout (no quantization); only the weights are
mxfp4. moe_gemm_a16w4 defaults to the gluon kernels; --backend pins one.

Everything except the GEMMs -- gating and routing -- is built once outside
the timed region, so the numbers cover the two projections and nothing else.
"""

import csv
import functools
import inspect
from itertools import chain
from pathlib import Path

import torch

from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.moe.moe_op_gemm_a16w4 import moe_gemm_a16w4
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.shuffle import shuffle_scale_moe
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


def check_and_shuffle_scales(scale, N, K):
    if N % 32 == 0 and K % (32 * 8) == 0:
        return shuffle_scale_moe(
            scale,
            arch=get_arch(),
            preshuffle_factor=32,
            scale_kwidth=8,
            return_layout=True,
        )
    else:
        return scale, None


def quantize(x, dtype):
    if dtype == "bf16":
        x = x.to(torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)
        return x, None
    elif dtype == "fp8":
        scale = x.abs().max().item() / 448.0
        fp8e4_dtype = (
            torch.float8_e4m3fn if get_arch() != "gfx942" else torch.float8_e4m3fnuz
        )
        x = x.to(fp8e4_dtype)
        return x, scale
    elif dtype == "mx8":
        fp8e4_dtype = (
            torch.float8_e4m3fn if get_arch() != "gfx942" else torch.float8_e4m3fnuz
        )
        x, scale = downcast_to_mxfp(x, fp8e4_dtype, axis=1)
        return x, scale
    else:
        assert dtype == "mx4", f"{dtype=}"
        x, scale = downcast_to_mxfp(x.to(torch.bfloat16), torch.uint8, axis=1)
        return x, scale


def _generate_mlp_pool_entry(
    dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP, backend, batch
):
    """Build one full weight-init + routing + activation set.

    Pure setup, not measured. Each call is one rotating-buffer pool entry:
    independent random weights and routing, so replaying through the pool
    during timing defeats cache residency the same way a repeat-and-average
    loop would, without the cost of a fresh CUDA graph capture each time.
    """
    dev = f"cuda:{torch.cuda.current_device()}"

    wg = torch.randn((dim1, n_expts_tot), device=dev)
    w1 = torch.randn((n_expts_tot, dim1, dim2 // TP), device=dev)
    w2 = torch.randn((n_expts_tot, dim2 // TP // 2, dim1), device=dev)
    bg = torch.randn((n_expts_tot,), device=dev)
    b1 = torch.randn((n_expts_tot, dim2 // TP), device=dev)
    b2 = torch.randn((n_expts_tot, dim1), device=dev)

    wg, _ = quantize(wg, "bf16")
    w1, w1_scale = quantize(w1, w_dtype)
    w2, w2_scale = quantize(w2, w_dtype)
    w1_scale, swizzle1 = check_and_shuffle_scales(w1_scale, dim2 // TP, dim1)
    w2_scale, swizzle2 = check_and_shuffle_scales(w2_scale, dim1, dim2 // TP // 2)

    x_dtype_torch = torch.bfloat16 if x_dtype == "bf16" else torch.float16
    x = torch.randn((batch, dim1), dtype=x_dtype_torch, device=dev)
    logits = gemm_a16w16(x, wg.T, bg)
    rdata, gather_indx, _scatter_indx = routing(logits, n_expts_act)

    y1 = moe_gemm_a16w4(
        x,
        w1,
        None,
        w1_scale,
        None,
        None,
        b1,
        rdata,
        gather_indx=gather_indx,
        swizzle_mx_scale=swizzle1,
        out_dtype=x_dtype_torch,
        apply_swiglu=True,
        backend=backend,
    )
    y1_bytes = y1.numel() * y1.element_size()

    y2 = moe_gemm_a16w4(
        y1,
        w2,
        None,
        w2_scale,
        None,
        None,
        b2,
        rdata,
        gather_indx=gather_indx,
        swizzle_mx_scale=swizzle2,
        out_dtype=x_dtype_torch,
        apply_swiglu=False,
        backend=backend,
    )

    # -- analytic FLOPs / bytes: 2*M*N*K per GEMM, and activations +
    # routed-expert weights + matmul output for traffic. mx scales (~1/16 of
    # the weight bytes) are not counted.
    n_tokens = gather_indx.shape[0]  # routed rows == batch * n_expts_act
    routed = int((rdata.expt_data.hist > 0).sum())  # experts that got >= 1 token
    elem = x.element_size()

    def w_bytes(w):
        return (w.numel() * w.element_size() // n_expts_tot) * routed

    def b_bytes(b):
        return (b.numel() * b.element_size() // n_expts_tot) * routed

    moe1_flops = 2 * n_tokens * (dim2 // TP) * dim1  # N = dim2 // TP, K = dim1
    moe1_bytes = x.numel() * elem + w_bytes(w1) + y1_bytes + b_bytes(b1)
    moe2_flops = 2 * n_tokens * dim1 * (dim2 // TP // 2)  # N = dim1, K = dim2/TP/2
    moe2_bytes = y1.numel() * elem + w_bytes(w2) + y2.numel() * elem + b_bytes(b2)

    meta = {
        "swizzle1": swizzle1,
        "swizzle2": swizzle2,
        "moe1_flops": moe1_flops,
        "moe1_bytes": moe1_bytes,
        "moe2_flops": moe2_flops,
        "moe2_bytes": moe2_bytes,
        "routed": routed,
        "block_m": rdata.block_m,
    }
    return (x, w1, w1_scale, b1, y1, w2, w2_scale, b2, rdata, gather_indx, meta)


_MLP_POOL_FIELDS = [
    "x",
    "w1",
    "w1_scale",
    "b1",
    "y1",
    "w2",
    "w2_scale",
    "b2",
    "rdata",
    "gather_indx",
    "meta",
]


def _make_moe_layer_fns(backend, swizzle1, swizzle2, out_dtype):
    """Parameterized moe1/moe2/total kernel callables for do_bench_aiter_triton.

    Argument names match a rotating-buffer pool entry's keys (``**_``
    absorbs the ones a given layer does not need).
    """

    def layer1(x, w1, w1_scale, b1, rdata, gather_indx, **_):
        moe_gemm_a16w4(
            x,
            w1,
            None,
            w1_scale,
            None,
            None,
            b1,
            rdata,
            gather_indx=gather_indx,
            swizzle_mx_scale=swizzle1,
            out_dtype=out_dtype,
            apply_swiglu=True,
            backend=backend,
        )

    def layer2(y1, w2, w2_scale, b2, rdata, gather_indx, **_):
        moe_gemm_a16w4(
            y1,
            w2,
            None,
            w2_scale,
            None,
            None,
            b2,
            rdata,
            gather_indx=gather_indx,
            swizzle_mx_scale=swizzle2,
            out_dtype=out_dtype,
            apply_swiglu=False,
            backend=backend,
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
    x_dtype,
    w_dtype,
    TP,
    backend=None,
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
        x_dtype,
        w_dtype,
        TP,
        backend,
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
    # the scale swizzle and block_m depend only on batch/topk/E, none of
    # which move with the weight draw
    swizzle1 = metas[0]["swizzle1"]
    swizzle2 = metas[0]["swizzle2"]
    block_m = metas[0]["block_m"]

    out_dtype = torch.bfloat16 if x_dtype == "bf16" else torch.float16
    layer1_fn, layer2_fn, both_fn = _make_moe_layer_fns(
        backend, swizzle1, swizzle2, out_dtype
    )

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
    x_dtype,
    w_dtype,
    TP,
    name="",
    layers=LAYERS,
    num_weight_inits=1,
    backend=None,
    kwargs_rot_bufs=None,
    kwargs_do_bench_graph=None,
):
    out_dir = Path("logs") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = (
        f"{x_dtype}x-{w_dtype}w-TP{TP}-dim1={dim1}-dim2={dim2}"
        f"-E={n_expts_tot}-topk={n_expts_act}"
    )
    if tuple(layers) != LAYERS:
        stem += "-layers=" + "+".join(layers)
    out_csv = out_dir / f"{stem}.csv"

    print(f"batch_sizes={batch_sizes}")
    compute_roofline(
        dim1,
        dim2,
        n_expts_tot,
        n_expts_act,
        x_dtype,
        w_dtype,
        TP,
        backend,
        layers,
        num_weight_inits,
        bench_fn=bench_mlp,
        intensity_proxy_name="batch",
        intensity_proxy_values=batch_sizes,
        out_path=out_csv,
        kwargs_rot_bufs=kwargs_rot_bufs,
        kwargs_do_bench_graph=kwargs_do_bench_graph,
    )


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="MoE")
    parser.set_defaults(backend="gluon")

    parser.add_argument(
        "--M",
        type=int,
        nargs="+",
        default=None,
        help="MoE batch sizes M (one or more integers). "
        "If not set, a predermined list of values will be used.",
    )
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
    parser.add_argument(
        "--TP",
        type=int,
        default=1,
        help="Tensor Parallelism(TP)",
    )
    return parser.parse_args(args=args)


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args=args)

    dim1, dim2 = parsed_args.shape
    total_experts, active_experts = parsed_args.experts
    if parsed_args.M is None:
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
    else:
        batch_sizes_moe = parsed_args.M

    quantized_dtypes = ["bf16", "mx4"]

    roofline_mlp(
        batch_sizes_moe,
        dim1,
        dim2,
        total_experts,
        active_experts,
        quantized_dtypes[0],
        quantized_dtypes[1],
        TP=parsed_args.TP,
        name="default-model",
        layers=tuple(n for n in LAYERS if n in parsed_args.layers),
        num_weight_inits=parsed_args.num_weight_inits,
        backend=parsed_args.backend,
        kwargs_rot_bufs=get_rot_bufs_kwargs(parsed_args),
        kwargs_do_bench_graph=get_do_bench_graph_kwargs(parsed_args),
    )


if __name__ == "__main__":
    main()
