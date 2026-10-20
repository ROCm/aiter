# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/bench/bench_mlp.py
"""Benchmark the fp8/blockscale-fp8 MoE MLP (two moe_gemm_a8w8_blockscale calls).

Each batch size is measured with `do_bench_aiter_triton` (a rotating-buffer
CUDA graph replay) and reported as three rows, keyed by the `layer` column:

  moe1   the gathered up projection alone   (N = dim2 / TP, K = dim1, + swiglu)
  moe2   the scattered down projection alone (N = dim1, K = dim2 / TP / 2)
  total  both back to back, i.e. the whole MLP

--act-dtype / --w-dtype each pick fp8 (e4m3 with one static scale per tensor)
or bs8 (block-scaled fp8, one fp32 scale per 128x128 tile). Both GEMMs emit
fp8 directly from out_dtype, so there is no separate quant kernel between
moe1 and moe2 -- the activation scale (static or per-block) is reused as-is
for both layers, matching the reference this benchmark is modeled on.

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
from aiter.ops.triton.moe.moe_op_gemm_a8w8_blockscale import (
    moe_gemm_a8w8_blockscale,
)
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.utils.gemm_config_utils import (
    compute_splitk_params,
    get_gemm_config,
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
ACT_DTYPES = ("fp8", "bs8")
_GROUP_SHAPE = (128, 128, 128)  # (group_shape_m, group_shape_n, group_shape_k)


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


def _num_blocks(length, block):
    return (length + block - 1) // block


def _generate_mlp_pool_entry(
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    x_dtype_str,
    w_dtype_str,
    per_row_act_quant,
    TP,
    batch,
):
    """Build one full weight-init + routing + activation set.

    Pure setup, not measured. Each call is one rotating-buffer pool entry:
    independent random weights/scales and routing, so replaying through the
    pool during timing defeats cache residency the same way a repeat-and-
    average loop would, without the cost of a fresh CUDA graph capture each
    time. Weights and scales are synthetic (random, not calibrated from real
    data), matching the reference this benchmark is modeled on -- this is a
    throughput benchmark, and GEMM cost does not depend on data values.
    """
    dev = f"cuda:{torch.cuda.current_device()}"
    group_m, group_n, group_k = _GROUP_SHAPE

    wg = torch.randn((dim1, n_expts_tot), dtype=torch.bfloat16, device=dev)
    w1 = (
        torch.randn((n_expts_tot, dim1, dim2 // TP), dtype=torch.bfloat16, device=dev)
        / 10
    ).to(torch.float8_e4m3fn)
    w2 = (
        torch.randn(
            (n_expts_tot, dim2 // TP // 2, dim1), dtype=torch.bfloat16, device=dev
        )
        / 10
    ).to(torch.float8_e4m3fn)
    bg = torch.randn((n_expts_tot,), device=dev)
    b1 = torch.randn((n_expts_tot, dim2 // TP), dtype=torch.float32, device=dev)
    b2 = torch.randn((n_expts_tot, dim1), dtype=torch.float32, device=dev)

    if x_dtype_str == "fp8":
        x_static_scale = torch.tensor(1e-4, device=dev)
        x_scale = None
    else:
        x_static_scale = None
        k_blocks_x = _num_blocks(dim1, group_k)
        if per_row_act_quant:
            x_scale = torch.rand((batch, k_blocks_x), dtype=torch.float32, device=dev)
        else:
            m_blocks = _num_blocks(batch, group_m)
            x_scale = torch.rand(
                (m_blocks, k_blocks_x), dtype=torch.float32, device=dev
            )

    if w_dtype_str == "fp8":
        w_static_scale = torch.tensor(1e-4, device=dev)
        w1_scale = w2_scale = None
    else:
        w_static_scale = None
        k_blocks_w1 = _num_blocks(dim1, group_k)
        n_blocks_w1 = _num_blocks(dim2 // TP, group_n)
        k_blocks_w2 = _num_blocks(dim2 // TP // 2, group_k)
        n_blocks_w2 = _num_blocks(dim1, group_n)
        w1_scale = torch.rand(
            (n_expts_tot, k_blocks_w1, n_blocks_w1), dtype=torch.float32, device=dev
        )
        w2_scale = torch.rand(
            (n_expts_tot, k_blocks_w2, n_blocks_w2), dtype=torch.float32, device=dev
        )

    x1 = (torch.randn((batch, dim1), dtype=torch.bfloat16, device=dev) / 10).to(
        torch.float8_e4m3fn
    )
    xg = x1.to(torch.bfloat16)

    M, K = xg.shape
    K, N = wg.shape
    config, _ = get_gemm_config("GEMM-A16W16", M, N, K)
    compute_splitk_params(config, K)
    config["BLOCK_SIZE_M"] = min(config["BLOCK_SIZE_M"], 128)
    config["BLOCK_SIZE_N"] = min(config["BLOCK_SIZE_N"], 128)
    config["BLOCK_SIZE_K"] = min(config["BLOCK_SIZE_K"], 128)
    logits = gemm_a16w16(xg, wg.T, bg, config=config)
    rdata, gather_indx, scatter_indx = routing(logits, n_expts_act)

    out_dtype = torch.float8_e4m3fn
    y1 = moe_gemm_a8w8_blockscale(
        x1,
        w1,
        x_scale,
        w1_scale,
        x_static_scale,
        w_static_scale,
        None,
        b1,
        rdata,
        gather_indx=gather_indx,
        out_dtype=out_dtype,
        apply_swiglu=True,
    )
    y2 = moe_gemm_a8w8_blockscale(
        y1,
        w2,
        x_scale,
        w2_scale,
        x_static_scale,
        w_static_scale,
        None,
        b2,
        rdata,
        scatter_indx=scatter_indx,
        out_dtype=out_dtype,
    )

    # -- analytic FLOPs / bytes: 2*M*N*K per GEMM, and activations +
    # routed-expert weights + matmul output for traffic. Blockscale scales
    # are counted; the moe2 scatter reduction is not (its runtime is inside
    # moe_gemm_a8w8_blockscale and so is inside the measurement).
    n_tokens = gather_indx.shape[0]  # routed rows == batch * n_expts_act
    routed = int((rdata.expt_data.hist > 0).sum())  # experts that got >= 1 token

    def w_bytes(w):
        return (w.numel() * w.element_size() // n_expts_tot) * routed

    def scale_bytes(scale):
        if scale is None:
            return 0
        return (scale.numel() * scale.element_size() // n_expts_tot) * routed

    def b_bytes(b):
        return (b.numel() * b.element_size() // n_expts_tot) * routed

    x_scale_bytes = 0 if x_scale is None else x_scale.numel() * x_scale.element_size()
    moe1_flops = 2 * n_tokens * (dim2 // TP) * dim1  # N = dim2 // TP, K = dim1
    moe1_bytes = (
        x1.numel() * x1.element_size()
        + x_scale_bytes
        + w_bytes(w1)
        + scale_bytes(w1_scale)
        + y1.numel() * y1.element_size()
        + b_bytes(b1)
    )
    moe2_flops = 2 * n_tokens * dim1 * (dim2 // TP // 2)  # N = dim1, K = dim2/TP/2
    moe2_bytes = (
        y1.numel() * y1.element_size()
        + x_scale_bytes
        + w_bytes(w2)
        + scale_bytes(w2_scale)
        + n_tokens * dim1 * y2.element_size()
        + b_bytes(b2)
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
        w1,
        w1_scale,
        b1,
        y1,
        w2,
        w2_scale,
        b2,
        x_scale,
        x_static_scale,
        w_static_scale,
        rdata,
        gather_indx,
        scatter_indx,
        meta,
    )


_MLP_POOL_FIELDS = [
    "x1",
    "w1",
    "w1_scale",
    "b1",
    "y1",
    "w2",
    "w2_scale",
    "b2",
    "x_scale",
    "x_static_scale",
    "w_static_scale",
    "rdata",
    "gather_indx",
    "scatter_indx",
    "meta",
]


def _make_moe_layer_fns(out_dtype):
    """Parameterized moe1/moe2/total kernel callables for do_bench_aiter_triton.

    Argument names match a rotating-buffer pool entry's keys (``**_``
    absorbs the ones a given layer does not need).
    """

    def layer1(
        x1,
        w1,
        w1_scale,
        b1,
        x_scale,
        x_static_scale,
        w_static_scale,
        rdata,
        gather_indx,
        **_,
    ):
        moe_gemm_a8w8_blockscale(
            x1,
            w1,
            x_scale,
            w1_scale,
            x_static_scale,
            w_static_scale,
            None,
            b1,
            rdata,
            gather_indx=gather_indx,
            out_dtype=out_dtype,
            apply_swiglu=True,
        )

    def layer2(
        y1,
        w2,
        w2_scale,
        b2,
        x_scale,
        x_static_scale,
        w_static_scale,
        rdata,
        scatter_indx,
        **_,
    ):
        moe_gemm_a8w8_blockscale(
            y1,
            w2,
            x_scale,
            w2_scale,
            x_static_scale,
            w_static_scale,
            None,
            b2,
            rdata,
            scatter_indx=scatter_indx,
            out_dtype=out_dtype,
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
    per_row_act_quant,
    TP,
    layers=LAYERS,
    num_weight_inits=1,
    kwargs_rot_bufs=None,
    kwargs_do_bench_graph=None,
):
    kwargs_rot_bufs = kwargs_rot_bufs or {}
    kwargs_do_bench_graph = kwargs_do_bench_graph or {}
    assert dim2 % TP == 0, f"{dim2=}, {TP=}, dim2 must be divisible by TP"
    assert x_dtype in ACT_DTYPES, f"{x_dtype=} must be one of {ACT_DTYPES}"
    assert w_dtype in ACT_DTYPES, f"{w_dtype=} must be one of {ACT_DTYPES}"
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
        per_row_act_quant,
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

    layer1_fn, layer2_fn, both_fn = _make_moe_layer_fns(torch.float8_e4m3fn)

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
    per_row_act_quant,
    TP,
    layers=LAYERS,
    num_weight_inits=1,
    name="",
    kwargs_rot_bufs=None,
    kwargs_do_bench_graph=None,
):
    out_dir = Path("logs") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{x_dtype}x-{w_dtype}w-TP{TP}-dim1={dim1}-dim2={dim2}-E={n_expts_tot}-topk={n_expts_act}"
    if tuple(layers) != LAYERS:
        stem += "-layers=" + "+".join(layers)
    out_csv = out_dir / f"{stem}.csv"

    compute_roofline(
        dim1,
        dim2,
        n_expts_tot,
        n_expts_act,
        x_dtype,
        w_dtype,
        per_row_act_quant,
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


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="MoE")

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
        "--act-dtype",
        choices=list(ACT_DTYPES),
        default="fp8",
        help="Activation dtype, fp8 or bs8.",
    )
    parser.add_argument(
        "--w-dtype",
        choices=list(ACT_DTYPES),
        default="fp8",
        help="Weight dtype, fp8 or bs8.",
    )
    parser.add_argument(
        "--act-per-row-bs",
        action="store_true",
        help="Use per-row blockscale instead of per-M-block if act-dtype is bs8.",
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

    roofline_mlp(
        batch_sizes_moe,
        dim1,
        dim2,
        total_experts,
        active_experts,
        parsed_args.act_dtype,
        parsed_args.w_dtype,
        parsed_args.act_per_row_bs,
        TP=1,
        layers=tuple(n for n in LAYERS if n in parsed_args.layers),
        num_weight_inits=parsed_args.num_weight_inits,
        name="gpt-oss-x2",
        kwargs_rot_bufs=get_rot_bufs_kwargs(parsed_args),
        kwargs_do_bench_graph=get_do_bench_graph_kwargs(parsed_args),
    )


if __name__ == "__main__":
    main()
