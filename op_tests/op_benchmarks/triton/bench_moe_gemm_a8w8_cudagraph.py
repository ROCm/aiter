# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/bench/bench_mlp.py
"""Benchmark the a8w8 MoE MLP (two moe_gemm_a8w8 calls), across all four
activation/weight dtype combinations.

Each batch size is measured with `RotatingBufferBench` (CUDA graph replay)
and reported as up to three rows, keyed by the `layer` column:

  moe1   the gathered up projection alone   (N = dim2 / TP, K = dim1, + swiglu)
  moe2   the scattered down projection alone (N = dim1, K = dim2 / TP / 2)
  total  both back to back, i.e. the whole MLP

`total` is not `moe1 + moe2`: an isolated projection replays the same kernel
over and over, so its weights stay cache-resident in a way the real layer does
not. Compare like with like.

--layers picks which of those are measured, e.g. `--layers moe1 moe2` to skip
the back-to-back run, or `--layers moe1` for one projection on its own. The
setup is unchanged either way (moe2 needs moe1's output, so layer 1 always
runs once outside the timed region), so a row means the same thing whichever
subset it came from.

--act-dtype / --w-dtype pick the activation / weight format independently, so
all four combinations are reachable:

  fp8   e4m3 with one static scale per tensor -- hardcoded to 1e-4, exactly as
        the original bench_moe_gemm_a8w8.py does (not calibrated from the
        data). The kernel multiplies the scale back into the accumulator.
  mx8   mxfp8, i.e. e4m3 plus one ue8m0 scale per 32 values along K.

Layer 1 always applies swiglu. Its output dtype follows the original exactly,
which is NOT symmetric across the four combos: it is fp8 whenever either side
is static fp8 (fp8/fp8, fp8/mx8, mx8/fp8), and only bf16 (the kernel's own
default) when both sides are mx8. So the mx8/fp8 combo forces an fp8
intermediate and then re-quantizes it to mx8 for layer 2 -- an extra
quantization the other combos don't pay. When the activation side is fp8,
layer 2 reuses layer 1's fp8 output and the very same static scale directly
(no separate quantize() call and no recalibration from layer 1's output
magnitude); when the activation side is mx8, layer 2's input is always
freshly quantized to mx8 from layer 1's output. This file only reproduces
that sequence -- it does not change or "fix" it.

Bias is always on (b1/b2), matching the original, which has no --bias flag.
moe_gemm_a8w8 itself takes no backend/preshuffle argument (it is a single
triton kernel), so unlike the a8w4 MoE benchmark there is no --backend /
--preshuffle dispatch surface here.

Everything except the GEMMs -- gating, routing, and the activation
quantization that feeds each layer -- is built once outside the timed region,
so the numbers cover the two projections and nothing else.
"""

import csv
import functools
import inspect
from itertools import chain
from pathlib import Path

import torch

from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.moe.moe_op_gemm_a8w8 import moe_gemm_a8w8
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.quant_moe import (
    downcast_to_mxfp,
    downcast_to_static_fp8,
    downcast_to_static_fp8_3d,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.shuffle import shuffle_scale_moe
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
# activation / weight formats --act-dtype / --w-dtype accept
ACT_DTYPES = ("fp8", "mx8")
W_DTYPES = ("fp8", "mx8")
# the original's hardcoded static scale (not calibrated from data -- see
# module docstring for why this is preserved as-is)
STATIC_SCALE = 1e-4


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


def check_and_shuffle_scales(scale, N, K):
    """mxfp8 scale swizzle for the gfx950 (CDNA4) scale layout.

    Unlike the a8w4 MoE benchmark's version of this helper, this is NOT
    arch-gated via get_arch() -- it mirrors bench_moe_gemm_a8w8.py exactly,
    which always targets the gfx950 swizzle when the shape allows it.
    """
    if N % 32 == 0 and K % (32 * 8) == 0:
        scale = shuffle_scale_moe(
            scale, arch="gfx950", preshuffle_factor=32, scale_kwidth=8
        )
        return scale, "CDNA4_SCALE"
    else:
        return scale, None


def fp8_dtype():
    """e4m3, in the flavour this arch's kernels take."""
    return torch.float8_e4m3fn if get_arch() != "gfx942" else torch.float8_e4m3fnuz


def quantize(x, dtype):
    """bf16 / mxfp8 tensors (a8w8 weights/activations are never mxfp4)."""
    if dtype == "bf16":
        x = x.to(torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)
        return x, None
    assert dtype == "mx8", f"{dtype=}"
    x, scale = downcast_to_mxfp(x.to(torch.bfloat16), fp8_dtype(), axis=1)
    return x, scale


_POOL_FIELDS = [
    "x1",
    "w1",
    "x1_scales",
    "w1_scales",
    "x_static_scale",
    "w_static_scale",
    "b1",
    "rdata",
    "gather_indx",
    "swizzle_mx_scale1",
    "x2",
    "w2",
    "x2_scales",
    "w2_scales",
    "x2_static_scale",
    "b2",
    "scatter_indx",
    "swizzle_mx_scale2",
    "y1_bytes",
]


def _generate_pool_inputs(
    batch, dim1, dim2, n_expts_tot, n_expts_act, x_dtype, w_dtype, TP
):
    """One full weight-init draw: weights, routing, and the layer-1 swiglu
    output layer 2 reads. Regenerated per pool entry, so replaying through the
    pool rotates through distinct weight/routing draws instead of keeping one
    draw's tensors cache-resident -- the role `num_weight_inits` used to play
    via a Python averaging loop in bench_moe_gemm_a8w8.py.
    """
    rank = 0
    dev = f"cuda:{rank}"

    # -- init data --
    wg = torch.randn((dim1, n_expts_tot), dtype=torch.float32, device=dev)
    w1 = torch.randn((n_expts_tot, dim1, dim2 // TP), dtype=torch.float32, device=dev)
    w2 = torch.randn(
        (n_expts_tot, dim2 // TP // 2, dim1), dtype=torch.float32, device=dev
    )
    bg = torch.randn((n_expts_tot,), dtype=torch.float32, device=dev)
    # bias is always on -- the original has no --bias flag
    b1 = torch.randn((n_expts_tot, dim2 // TP), dtype=torch.float32, device=dev)
    b2 = torch.randn((n_expts_tot, dim1), dtype=torch.float32, device=dev)

    wg, _ = quantize(wg, "bf16")

    static_fp8_w = w_dtype == "fp8"
    if static_fp8_w:
        w_static_scale = torch.tensor(STATIC_SCALE, device=dev, dtype=torch.float32)
        w1 = downcast_to_static_fp8_3d(w1, w_static_scale)
        w2 = downcast_to_static_fp8_3d(w2, w_static_scale)
        w1_scales = w2_scales = None
        swizzle_mx_scale1 = swizzle_mx_scale2 = "CDNA4_SCALE"
    else:
        w_static_scale = None
        w1, w1_scales = quantize(w1, "mx8")
        w2, w2_scales = quantize(w2, "mx8")
        w1_scales, swizzle_mx_scale1 = check_and_shuffle_scales(
            w1_scales, dim2 // TP, dim1
        )
        w2_scales, swizzle_mx_scale2 = check_and_shuffle_scales(
            w2_scales, dim1, dim2 // TP // 2
        )

    # -- routing --
    x = torch.randn((batch, dim1), dtype=torch.bfloat16, device=dev)
    logits = gemm_a16w16(x, wg.T, bg)
    rdata, gather_indx, scatter_indx = routing(logits, n_expts_act)

    static_fp8_x = x_dtype == "fp8"
    if static_fp8_x:
        x_static_scale = torch.tensor(STATIC_SCALE, device=dev, dtype=torch.float32)
        x1 = downcast_to_static_fp8(x, x_static_scale)
        x1_scales = None
    else:
        x_static_scale = None
        x1, x1_scales = quantize(x, "mx8")

    # Layer 1's output dtype follows the original exactly: fp8 whenever
    # either side is static fp8, bf16 (the kernel default) only when both
    # sides are mx8 -- see the module docstring for why this is asymmetric.
    out_dtype1 = (
        torch.bfloat16 if not static_fp8_x and not static_fp8_w else fp8_dtype()
    )

    # layer 2 reads layer 1's swiglu output; get it (and, for mx8, its
    # scales) once here, outside the timed region. This doubles as the
    # compile warmup.
    y1 = moe_gemm_a8w8(
        x1,
        w1,
        x1_scales,
        w1_scales,
        x_static_scale,
        w_static_scale,
        None,
        b1,
        routing_data=rdata,
        gather_indx=gather_indx,
        swizzle_mx_scale=swizzle_mx_scale1,
        out_dtype=out_dtype1,
        apply_swiglu=True,
    )
    y1_bytes = y1.numel() * y1.element_size()

    if static_fp8_x:
        # layer 1 already emitted fp8 directly (out_dtype1 == fp8); reuse it
        # and the same static scale for layer 2, exactly as the original does
        # -- NOT recalibrated from y1's magnitude.
        x2, x2_scales, x2_static_scale = y1, None, x_static_scale
    else:
        # mx8 activations always get a fresh mx8 quantization before layer 2,
        # whether layer 1's output was fp8 (mx8/fp8 combo) or bf16 (mx8/mx8).
        x2, x2_scales = quantize(y1, "mx8")
        x2_static_scale = None

    return (
        x1,
        w1,
        x1_scales,
        w1_scales,
        x_static_scale,
        w_static_scale,
        b1,
        rdata,
        gather_indx,
        swizzle_mx_scale1,
        x2,
        w2,
        x2_scales,
        w2_scales,
        x2_static_scale,
        b2,
        scatter_indx,
        swizzle_mx_scale2,
        y1_bytes,
    )


def _layer1_fn(out_dtype1):
    def _fn(
        x1,
        w1,
        x1_scales,
        w1_scales,
        x_static_scale,
        w_static_scale,
        b1,
        rdata,
        gather_indx,
        swizzle_mx_scale1,
        **_,
    ):
        return moe_gemm_a8w8(
            x1,
            w1,
            x1_scales,
            w1_scales,
            x_static_scale,
            w_static_scale,
            None,
            b1,
            routing_data=rdata,
            gather_indx=gather_indx,
            swizzle_mx_scale=swizzle_mx_scale1,
            out_dtype=out_dtype1,
            apply_swiglu=True,
        )

    return _fn


def _layer2_fn():
    def _fn(
        x2,
        w2,
        x2_scales,
        w2_scales,
        x2_static_scale,
        w_static_scale,
        b2,
        rdata,
        scatter_indx,
        swizzle_mx_scale2,
        **_,
    ):
        return moe_gemm_a8w8(
            x2,
            w2,
            x2_scales,
            w2_scales,
            x2_static_scale,
            w_static_scale,
            None,
            b2,
            routing_data=rdata,
            scatter_indx=scatter_indx,
            swizzle_mx_scale=swizzle_mx_scale2,
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
    TP,
    layers=LAYERS,
    num_weight_inits=1,
):
    assert dim2 % TP == 0, f"{dim2=}, {TP=}, dim2 must be divisible by TP"
    assert x_dtype in ACT_DTYPES, f"{x_dtype=} must be one of {ACT_DTYPES}"
    assert w_dtype in W_DTYPES, f"{w_dtype=} must be one of {W_DTYPES}"
    assert layers, "at least one layer must be selected"
    assert set(layers) <= set(LAYERS), f"unknown layer(s) in {layers=}"

    static_fp8_x = x_dtype == "fp8"
    static_fp8_w = w_dtype == "fp8"
    out_dtype1 = (
        torch.bfloat16 if not static_fp8_x and not static_fp8_w else fp8_dtype()
    )

    _p_gen = functools.partial(
        _generate_pool_inputs,
        batch,
        dim1,
        dim2,
        n_expts_tot,
        n_expts_act,
        x_dtype,
        w_dtype,
        TP,
    )
    # pool_len == num_weight_inits: each captured call in the graph replays a
    # distinct weight/routing draw, the same role the old averaging loop played.
    rot_bufs = bencher.build_pool(
        [_p_gen],
        [_POOL_FIELDS],
        min_pool_len=num_weight_inits,
        max_pool_len=num_weight_inits,
    )

    layer1_fn = _layer1_fn(out_dtype1)
    layer2_fn = _layer2_fn()
    both_fn = _both_fn(layer1_fn, layer2_fn)

    # -- analytic FLOPs / bytes, matching the proton metadata the original's
    # kernel itself reports: 2*M*N*K per GEMM, and activations + routed-expert
    # weights + matmul output for traffic. mx scales and the moe2 scatter
    # reduction's own traffic are not counted (the reduction's runtime is
    # inside moe_gemm_a8w8 and so is inside the measurement). Taken from entry
    # 0: the shapes are identical across entries even though the routed count
    # can drift slightly with the random draw.
    sample = rot_bufs[0]
    n_tokens = sample["gather_indx"].shape[0]  # routed rows == batch * n_expts_act
    routed = int((sample["rdata"].expt_data.hist > 0).sum())  # experts w/ >= 1 token

    def w_bytes(w):
        return (w.numel() * w.element_size() // n_expts_tot) * routed

    def w_scale_bytes(w):
        # a8w8 weights are never packed (1 byte/element whether static fp8 or
        # mx8), unlike a8w4's mxfp4 weights -- no *2 unpack factor needed here.
        return (w.numel() * w.element_size() // 32 // n_expts_tot) * routed

    x1, w1, x2, w2 = sample["x1"], sample["w1"], sample["x2"], sample["w2"]
    b1, b2 = sample["b1"], sample["b2"]

    moe1_flops = 2 * n_tokens * (dim2 // TP) * dim1  # N = dim2 // TP, K = dim1
    moe1_bytes = (
        x1.numel() * x1.element_size()
        + w_bytes(w1)
        + sample["y1_bytes"]
        + (b1.numel() * b1.element_size() // n_expts_tot) * routed
    )
    if not static_fp8_x:
        moe1_bytes += x1.numel() // 32
    if not static_fp8_w:
        moe1_bytes += w_scale_bytes(w1)

    moe2_flops = 2 * n_tokens * dim1 * (dim2 // TP // 2)  # N = dim1, K = dim2/TP/2
    # y2 is the scatter-compressed [batch, dim1] result; the GEMM writes the
    # uncompressed [n_tokens, dim1] rows the reduction then combines. Computed
    # once here (outside the timed region) purely for its dtype/element size.
    y2 = layer2_fn(**sample)
    moe2_bytes = (
        x2.numel() * x2.element_size()
        + w_bytes(w2)
        + n_tokens * dim1 * y2.element_size()
        + (b2.numel() * b2.element_size() // n_expts_tot) * routed
    )
    if not static_fp8_x:
        moe2_bytes += x2.numel() // 32
    if not static_fp8_w:
        moe2_bytes += w_scale_bytes(w2)

    # moe_gemm_a8w8 is a single triton kernel -- no gluon/backend dispatch
    # like moe_gemm_a8w4, so the kernel name is constant across layers.
    kernel = "_moe_gemm_a8w8"
    kernels = {"moe1": kernel, "moe2": kernel, "total": kernel}

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
        choices=list(ACT_DTYPES),
        default="fp8",
        help="Activation format: fp8 (e4m3 with one static scale per tensor, "
        "hardcoded to 1e-4 as in the original) or mx8 (mxfp8, one ue8m0 "
        "scale per 32 values along K). Default: fp8.",
    )
    parser.add_argument(
        "--w-dtype",
        choices=list(W_DTYPES),
        default="fp8",
        help="Weight format: fp8 (e4m3 with one static scale per tensor, "
        "hardcoded to 1e-4 as in the original) or mx8 (mxfp8, one ue8m0 "
        "scale per 32 values along K). Default: fp8.",
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
        help="Number of different weight initializations to run for more stable results (default: 1). "
        "Use higher values (e.g., 10) for more stable benchmarks.",
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
        TP=1,
        # dedupe, keeping the canonical report order rather than argv order
        layers=tuple(n for n in LAYERS if n in parsed_args.layers),
        num_weight_inits=parsed_args.num_weight_inits,
        name="moe_gemm_a8w8",
    )


if __name__ == "__main__":
    main()
