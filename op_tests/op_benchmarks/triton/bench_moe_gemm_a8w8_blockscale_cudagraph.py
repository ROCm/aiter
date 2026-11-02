# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/bench/bench_mlp.py
"""Benchmark the fp8 MoE MLP with blockscale-capable scaling (two
moe_gemm_a8w8_blockscale calls).

Each batch size is measured with `RotatingBufferBench` (CUDA graph replay) and
reported as up to three rows, keyed by the `layer` column:

  moe1   the gathered up projection alone   (N = dim2 / TP, K = dim1, + swiglu)
  moe2   the scattered down projection alone (N = dim1, K = dim2 / TP / 2)
  total  both back to back, i.e. the whole MLP

`total` is not `moe1 + moe2`: an isolated projection replays the same kernel
over and over, so its weights stay cache-resident in a way the real layer does
not. Compare like with like.

--layers picks which of those are measured, e.g. `--layers moe1 moe2` to skip
the back-to-back run, or `--layers moe1` for one projection on its own. The
setup is unchanged either way (moe2 needs moe1's output, so layer 1 always runs
once outside the timed region), so a row means the same thing whichever subset
it came from.

Unlike the mxfp4 (a8w4) variant, both activations AND weights here are always
stored as plain e4m3 fp8 values -- `--act-dtype`/`--w-dtype` (each one of
"fp8"/"bs8") only pick which SCALE the kernel is given for that operand, not a
different storage format:

  fp8   one static scale for the whole tensor (a scalar, calibrated once up
        front and reused for both layer 1 and layer 2 -- same as the original
        script this was converted from).
  bs8   a per-block scale tensor, shaped from `group_shape` (M/N/K block sizes
        for blockwise scaling) via a local `num_blocks()` helper. For
        activations, `--act-per-row-bs` picks one scale row per token (True)
        vs one row per `group_shape[0]`-sized block of tokens (False).

All four (x_dtype, w_dtype) combinations are selectable independently via
`--act-dtype`/`--w-dtype`, matching the kernel's four call-signature branches
(`x_block_scales`/`x_static_scale` and `w_block_scales`/`w_static_scale` are
mutually exclusive pairs per operand).

The gate GEMM (`gemm_a16w16`) uses a real tuned config, not the default:
`get_gemm_config("GEMM-A16W16", M, N, K)` + `compute_splitk_params`, with
BLOCK_SIZE_M/N/K clamped to <= 128 -- carried over unchanged from the original
proton-based script.

Everything except the two GEMMs -- weight/bias init, gating, routing, and
picking the scale tensors -- is built once per weight-init draw, outside the
timed region (`_generate_pool_inputs`); layer 1 also runs once there (untimed)
so layer 2's input is ready and the first-call compile cost is paid during
setup, not during timing.

This file is the RotatingBufferBench/CUDA-graph sibling of
`bench_moe_gemm_a8w8_blockscale.py`, which profiles a tight, uncaptured Python
loop with `triton.profiler` (proton) instead and filters the resulting trace by
kernel-name regex. That approach cannot be ported into a CUDA-graph capture (the
loop recomputes the gate GEMM + routing every rep and relies on proton's
per-launch hook), so the original is left untouched and this is a new,
parallel script instead. FLOPs/bytes here are computed analytically from
tensor shapes -- not via any profiling/trace mechanism -- deliberately matching
the real kernel's own proton `matmul_launch_metadata` formula (routed-expert-
scaled weight bytes, gathered-row activation bytes, no scale/bias bytes
counted), so the numbers line up with the original script's CSV output for the
same shape.
"""

import csv
import functools
import inspect
from itertools import chain
from pathlib import Path

import torch

from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.moe.moe_op_gemm_a8w8_blockscale import moe_gemm_a8w8_blockscale
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.gemm_config_utils import (
    compute_splitk_params,
    get_gemm_config,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)

# measurable layers, in report order; see the module docstring
LAYERS = ("moe1", "moe2", "total")
# scale formats --act-dtype/--w-dtype accept; storage is always e4m3 fp8, see docstring
SCALE_DTYPES = ("fp8", "bs8")

# Default group_m, group_n, group_k (blockwise-scale block sizes)
group_shape = (128, 128, 128)


def compute_roofline(
    *args, bench_fn, intensity_proxy_name, intensity_proxy_values, out_path, **kwargs
):
    # validate input args
    if not isinstance(intensity_proxy_name, str):
        raise TypeError(
            "intensity_proxy must be a string naming a parameter in target_fn"
        )
    # determine position of intensity_proxy in target_fn signature
    sig = inspect.signature(bench_fn)
    params = list(sig.parameters.values())
    if intensity_proxy_name not in sig.parameters:
        raise ValueError(
            f"Parameter '{intensity_proxy_name}' not found in {bench_fn.__name__} signature"
        )
    pos_index = [p.name for p in params].index(intensity_proxy_name)

    # wrapper to inject intensity proxy into target_fn and call it
    def inject_proxy_and_call(val, args, kwargs):
        args_list = list(args)
        args_list.insert(pos_index, val)
        return bench_fn(*args_list, **kwargs)

    # collect performance data
    perfs = []
    print("=========================================")
    print(f"{out_path}...")
    print("=========================================")

    for val in intensity_proxy_values:
        perf = inject_proxy_and_call(val, args, kwargs)
        perfs.append((val, perf))

        # one line per value, one "<layer> <us> <TFLOP/s> <TB/s> [<kernel>]"
        # group per measurement -- the same numbers the CSV carries per row.
        groups = " | ".join(
            f"{name} {lp['latency_ms'] * 1e3:.2f}us "
            f"{lp['flops'] / lp['latency_ms'] * 1e-9:#.4g} TF/s "
            f"{lp['bytes'] / lp['latency_ms'] * 1e-9:#.4g} TB/s "
            f"[{lp['kernel']}]"
            for name, lp in perf["layers"].items()
        )
        print(
            f"{intensity_proxy_name}: {val:5d} | {groups} | "
            f"block_m={perf['block_m']} routed_experts={perf['routed_experts']}"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # long format: one row per (value, layer), so a sweep stays easy to group
    fieldnames = [
        intensity_proxy_name,  # e.g. "batch"
        "layer",
        "latency_us",
        "tflops",
        "tbps",
        "flops",
        "bytes",
        "kernel",
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
                        "kernel": lp["kernel"],
                        "block_m": perf["block_m"],
                        "routed_experts": perf["routed_experts"],
                    }
                )


def out_dtype_for_arch():
    """e4m3, in the flavour this arch's kernels take (uz on gfx942)."""
    return torch.float8_e4m3fn if get_arch() != "gfx942" else torch.float8_e4m3fnuz


def num_blocks(length, block):
    return (length + block - 1) // block


_POOL_FIELDS = [
    "x1",
    "w1",
    "w2",
    "b1",
    "b2",
    "rdata",
    "gather_indx",
    "scatter_indx",
    "x_scale",
    "x_static_scale",
    "w1_scale",
    "w2_scale",
    "w_static_scale",
    "x2",
]


def _generate_pool_inputs(
    batch,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    x_dtype_str,
    w_dtype_str,
    per_row_act_quant,
    TP,
    out_dtype,
):
    """One full weight-init draw: weights, biases, routing, the scale tensors
    for the requested (x_dtype_str, w_dtype_str) combo, and the layer-1 output
    layer 2 reads. Regenerated per pool entry, so replaying through the pool
    rotates through distinct weight/routing draws instead of keeping one
    draw's tensors cache-resident -- the role `num_weight_inits` used to play
    via the original script's Python averaging loop.
    """
    rank = 0
    dev = f"cuda:{rank}"

    group_shape_m, group_shape_n, group_shape_k = group_shape

    # -- init data -- weights/activations are always plain e4m3 fp8; only the
    # accompanying scale tensors differ by combo (see module docstring).
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

    bg = torch.randn((n_expts_tot,), dtype=torch.float32, device=dev)
    b1 = torch.randn((n_expts_tot, dim2 // TP), dtype=torch.float32, device=dev)
    b2 = torch.randn((n_expts_tot, dim1), dtype=torch.float32, device=dev)

    x1 = (torch.randn((batch, dim1), dtype=torch.bfloat16, device=dev) / 10).to(
        torch.float8_e4m3fn
    )
    xg = x1.to(torch.bfloat16)

    # -- scales: static (one scalar, shared by both layers) or blockwise (one
    # tensor per operand per layer), matching the original script exactly --
    if x_dtype_str == "fp8":
        x_static_scale = torch.tensor(1e-4, dtype=torch.float32, device=dev)
        x_scale = None
    else:
        x_static_scale = None
        k_blocks_x = num_blocks(dim1, group_shape_k)
        if per_row_act_quant:
            x_scale = torch.rand((batch, k_blocks_x), dtype=torch.float32, device=dev)
        else:
            m_blocks = num_blocks(batch, group_shape_m)
            x_scale = torch.rand(
                (m_blocks, k_blocks_x), dtype=torch.float32, device=dev
            )

    if w_dtype_str == "fp8":
        w_static_scale = torch.tensor(1e-4, dtype=torch.float32, device=dev)
        w1_scale = w2_scale = None
    else:
        w_static_scale = None
        k_blocks_w1 = num_blocks(dim1, group_shape_k)
        n_blocks_w1 = num_blocks(dim2 // TP, group_shape_n)
        k_blocks_w2 = num_blocks(dim2 // TP // 2, group_shape_k)
        n_blocks_w2 = num_blocks(dim1, group_shape_n)
        w1_scale = torch.rand(
            (n_expts_tot, k_blocks_w1, n_blocks_w1), dtype=torch.float32, device=dev
        )
        w2_scale = torch.rand(
            (n_expts_tot, k_blocks_w2, n_blocks_w2), dtype=torch.float32, device=dev
        )

    # -- gate GEMM with a real tuned config (not the default), + routing --
    M, _K = xg.shape
    K2, N = wg.shape
    config, _ = get_gemm_config("GEMM-A16W16", M, N, K2)
    compute_splitk_params(config, K2)
    config["BLOCK_SIZE_M"] = min(config["BLOCK_SIZE_M"], 128)
    config["BLOCK_SIZE_N"] = min(config["BLOCK_SIZE_N"], 128)
    config["BLOCK_SIZE_K"] = min(config["BLOCK_SIZE_K"], 128)

    logits = gemm_a16w16(xg, wg.T, bg, config=config)
    rdata, gather_indx, scatter_indx = routing(logits, n_expts_act)

    # layer 2 reads layer 1's swiglu output; get it once here, outside the
    # timed region. This doubles as the compile warmup. The kernel quantizes
    # its own output to out_dtype in the epilogue, so x2 needs no separate
    # quant step.
    x2 = moe_gemm_a8w8_blockscale(
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

    return (
        x1,
        w1,
        w2,
        b1,
        b2,
        rdata,
        gather_indx,
        scatter_indx,
        x_scale,
        x_static_scale,
        w1_scale,
        w2_scale,
        w_static_scale,
        x2,
    )


def _layer1_fn(out_dtype):
    def _fn(
        x1,
        w1,
        b1,
        rdata,
        gather_indx,
        x_scale,
        x_static_scale,
        w1_scale,
        w_static_scale,
        **_,
    ):
        return moe_gemm_a8w8_blockscale(
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

    return _fn


def _layer2_fn(out_dtype):
    def _fn(
        x2,
        w2,
        b2,
        rdata,
        scatter_indx,
        x_scale,
        x_static_scale,
        w2_scale,
        w_static_scale,
        **_,
    ):
        return moe_gemm_a8w8_blockscale(
            x2,
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

    return _fn


def _both_fn(layer1_fn, layer2_fn):
    def _fn(**kwargs):
        layer1_fn(**kwargs)
        layer2_fn(**kwargs)

    return _fn


def bench_mlp(
    bencher,
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
):
    assert dim2 % TP == 0, f"{dim2=}, {TP=}, dim2 must be divisible by TP"
    assert x_dtype in SCALE_DTYPES, f"{x_dtype=} must be one of {SCALE_DTYPES}"
    assert w_dtype in SCALE_DTYPES, f"{w_dtype=} must be one of {SCALE_DTYPES}"
    assert layers, "at least one layer must be selected"
    assert set(layers) <= set(LAYERS), f"unknown layer(s) in {layers=}"

    out_dtype = out_dtype_for_arch()

    _p_gen = functools.partial(
        _generate_pool_inputs,
        batch,
        dim1,
        dim2,
        n_expts_tot,
        n_expts_act,
        x_dtype,
        w_dtype,
        per_row_act_quant,
        TP,
        out_dtype,
    )
    # pool_len == num_weight_inits: each captured call in the graph replays a
    # distinct weight/routing draw, the same role the old averaging loop played.
    rot_bufs = bencher.build_pool(
        [_p_gen],
        [_POOL_FIELDS],
        min_pool_len=num_weight_inits,
        max_pool_len=num_weight_inits,
    )

    layer1_fn = _layer1_fn(out_dtype)
    layer2_fn = _layer2_fn(out_dtype)
    both_fn = _both_fn(layer1_fn, layer2_fn)

    # -- analytic FLOPs / bytes, matching the proton metadata the real kernel
    # reports (matmul_launch_metadata in
    # aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w8_blockscale.py):
    # 2*n_tokens*N*K flops, and x/y/w traffic with NO scale or bias bytes
    # counted (the kernel's own metric does not count them either). Weight
    # bytes are scaled down to the experts that actually received >= 1 routed
    # token, same as the kernel's own `(hist > 0).sum()`. Taken from entry 0:
    # shapes are identical across entries even though the routed count can
    # drift slightly with the random draw.
    sample = rot_bufs[0]
    n_tokens = sample["gather_indx"].shape[0]  # routed rows == batch * n_expts_act
    routed = int((sample["rdata"].expt_data.hist > 0).sum())  # experts w/ >= 1 token

    def w_bytes(w):
        return (w.numel() * w.element_size() // n_expts_tot) * routed

    x1, w1, x2, w2 = sample["x1"], sample["w1"], sample["x2"], sample["w2"]
    moe1_flops = 2 * n_tokens * (dim2 // TP) * dim1  # N = dim2 // TP, K = dim1
    moe1_bytes = (
        x1.numel() * x1.element_size() + w_bytes(w1) + x2.numel() * x2.element_size()
    )

    moe2_flops = 2 * n_tokens * dim1 * (dim2 // TP // 2)  # N = dim1, K = dim2/TP/2
    # y2 is the scatter-compressed [batch, dim1] result; the GEMM writes the
    # uncompressed [n_tokens, dim1] rows the reduction then combines. Computed
    # once here (outside the timed region) purely for its dtype/element size.
    y2 = layer2_fn(**sample)
    moe2_bytes = (
        x2.numel() * x2.element_size()
        + w_bytes(w2)
        + n_tokens * dim1 * y2.element_size()
    )

    kernels = {
        "moe1": "_moe_gemm_a8w8_blockscale_layer1",
        "moe2": "_moe_gemm_a8w8_blockscale_layer2",
        "total": "_moe_gemm_a8w8_blockscale",
    }

    # -- benchmark: each projection on its own, then the pair back to back,
    # keeping only what `layers` asked for (in LAYERS order, not argv order).
    # `total` is NOT moe1 + moe2 -- an isolated projection replays one kernel
    # over and over, so its weights stay hotter than they are in the real layer.
    to_bench = {
        "moe1": (layer1_fn, moe1_flops, moe1_bytes),
        "moe2": (layer2_fn, moe2_flops, moe2_bytes),
        "total": (both_fn, moe1_flops + moe2_flops, moe1_bytes + moe2_bytes),
    }
    measured = {
        name: {
            "latency_ms": bencher.bench(f),
            "flops": flops,
            "bytes": byts,
            "kernel": kernels[name],
        }
        for name, (f, flops, byts) in to_bench.items()
        if name in layers
    }

    return {
        "layers": measured,
        "block_m": sample["rdata"].block_m,
        "routed_experts": routed,
    }


def roofline_mlp(
    bencher,
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
):
    # Put all outputs under logs/<name>/ and write a CSV file (not a directory-as-stem).
    out_dir = Path("logs") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Every knob that changes what is measured goes in the filename, so sweeps
    # over different shapes/dtypes land side by side instead of overwriting.
    stem = (
        f"{x_dtype}x-{w_dtype}w-TP{TP}-dim1={dim1}-dim2={dim2}"
        f"-E={n_expts_tot}-topk={n_expts_act}"
    )
    if tuple(layers) != LAYERS:
        # a partial run holds a subset of the rows, so give it its own file
        stem += "-layers=" + "+".join(layers)
    out_csv = out_dir / f"{stem}.csv"

    compute_roofline(
        bencher,
        dim1,
        dim2,
        n_expts_tot,
        n_expts_act,
        x_dtype,
        w_dtype,
        per_row_act_quant,
        TP,  # fixed args
        layers,
        num_weight_inits,
        bench_fn=bench_mlp,  # function to benchmark
        intensity_proxy_name="batch",  # intensity proxy name
        intensity_proxy_values=batch_sizes,  # intensity proxy values to sweep
        out_path=out_csv,  # output path
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
        nargs=2,
        required=True,
        metavar=("DIM1", "DIM2"),
        help="The two MLP feature dimensions. DIM1 is the model (hidden) dim, "
        "i.e. the width of a token vector going into and coming out of the "
        "layer. DIM2 is the gated up-projection width -- twice the FFN "
        "intermediate size, since swiglu halves it. Together they fix both "
        "GEMMs: moe1 is N=DIM2/TP, K=DIM1, and moe2 is N=DIM1, K=DIM2/TP/2.",
    )
    parser.add_argument(
        "--experts",
        type=int,
        nargs=2,
        required=True,
        metavar=("TOTAL", "TOPK"),
        help="TOTAL is how many experts the layer holds; TOPK is how many of "
        "them each token is routed to. TOTAL sets the weight tensors' expert "
        "dim, TOPK multiplies the row count each GEMM sees (batch * TOPK "
        "routed rows).",
    )
    parser.add_argument(
        "--act-dtype",
        choices=list(SCALE_DTYPES),
        default="fp8",
        help="Activation scale format: fp8 (one static scale for the whole "
        "tensor) or bs8 (a per-block scale tensor, blockwise per "
        "--act-per-row-bs). Storage is e4m3 fp8 either way. Default: fp8.",
    )
    parser.add_argument(
        "--w-dtype",
        choices=list(SCALE_DTYPES),
        default="fp8",
        help="Weight scale format: fp8 (one static scale for the whole "
        "tensor, shared by both layers) or bs8 (a per-block scale tensor per "
        "layer). Storage is e4m3 fp8 either way. Default: fp8.",
    )
    parser.add_argument(
        "--act-per-row-bs",
        type=str,
        default="False",
        help="Use per-row blockscale (True) or per-M-block (False) if "
        "--act-dtype is bs8. NOTE: compared as a string ('True'/'False'), not "
        "parsed as a real bool -- kept exactly as the original script's CLI "
        "behavior.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        choices=LAYERS,
        default=list(LAYERS),
        help="Which layers to measure: moe1 (up projection), moe2 (down "
        "projection), total (both back to back). E.g. '--layers moe1 moe2' to "
        "skip the back-to-back run, or '--layers moe1' for one projection "
        "alone. Default: all three.",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=20,
        help="Unused: RotatingBufferBench.bench's replay count is a fixed "
        "constant, not an adaptive rep target. Kept for CLI compatibility.",
    )
    parser.add_argument(
        "--num-weight-inits",
        type=int,
        default=1,
        help="Number of different weight initializations to run for more "
        "stable results (default: 1). Use higher values (e.g., 10) for more "
        "stable benchmarks.",
    )
    args = parser.parse_args(args=args)
    return args


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

    # NOTE: preserved verbatim from the original script -- this is a string
    # comparison, not a real bool, and --act-per-row-bs is likewise a str flag.
    per_row_act_quant = parsed_args.act_per_row_bs == "True"

    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(parsed_args), **get_do_bench_graph_kwargs(parsed_args)}
    )

    roofline_mlp(
        bencher,
        batch_sizes_moe,
        dim1,
        dim2,
        total_experts,
        active_experts,
        parsed_args.act_dtype,
        parsed_args.w_dtype,
        per_row_act_quant,
        TP=1,
        # dedupe, keeping the canonical report order rather than argv order
        layers=tuple(n for n in LAYERS if n in parsed_args.layers),
        num_weight_inits=parsed_args.num_weight_inits,
        name="gpt-oss-x2-cudagraph",
    )


if __name__ == "__main__":
    main()
