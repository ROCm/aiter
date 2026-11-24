# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/bench/bench_mlp.py
"""Benchmark the bf16/fp16 x mxfp4 MoE MLP (two moe_gemm_a16w4 calls).

Each batch size is measured with `RotatingBufferBench` (CUDA graph replay)
and reported as up to three rows, keyed by the `layer` column:

  moe1   the gathered up projection alone    (N = dim2 / TP, K = dim1, + swiglu)
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

--act-dtype picks the activation dtype: bf16 or fp16. Unlike moe_gemm_a8w4,
moe_gemm_a16w4 never quantizes the activations -- they are plain 16-bit
tensors end to end (moe_gemm_a16w4 hard-asserts x_scales/x_static_scale/
quant_static_scale are all None). Only the weights are mxfp4 (the w4 half of
a16w4). Layer 1's swiglu output is handed to layer 2 completely as-is, with no
quantization step in between -- there is nothing to calibrate a scale from,
unlike the a8w4 benchmark.

Bias is always applied (b1/b2 are created and passed on every call); the
original bench_moe_gemm_a16w4.py has no --bias toggle, so this script doesn't
add one either.

Everything except the GEMMs -- gating, routing, and handing layer 1's output to
layer 2 -- is built once outside the timed region, so the numbers cover the two
projections and nothing else.

moe_gemm_a16w4 defaults to the gluon kernels on gfx1250 (this file preserves
the original's `--backend` default of "gluon") and falls back to triton on
other archs, or on gfx950 when K is not a multiple of 256 or scales aren't
swizzled. --backend pins one instead.
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
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)

# measurable layers, in report order; see the module docstring
LAYERS = ("moe1", "moe2", "total")
# activation formats --act-dtype accepts; both unquantized, the weights are
# always mxfp4
ACT_DTYPES = ("bf16", "fp16")


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
        # The kernel sits in the group because moe1 and moe2 can dispatch to
        # different variants of the same op (see the module docstring).
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
    """a16w4's own scale-shuffle gate (re-derived from bench_moe_gemm_a16w4.py,
    not copied from the a8w4 benchmark): arch-agnostic, via
    shuffle_scale_moe(..., return_layout=True)."""
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
    """bf16 passthrough / mxfp4 weight quantization.

    a16w4's activations are never quantized (see module docstring), so the
    only quantized values produced here are the mxfp4 weights; the "bf16"
    branch is only used for the untimed gating weight.
    """
    if dtype == "bf16":
        x = x.to(torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)
        return x, None
    else:
        assert dtype == "mx4", f"{dtype=}"
        x, scale = downcast_to_mxfp(x.to(torch.bfloat16), torch.uint8, axis=1)
        return x, scale


def backend_name(backend=None):
    """Backend moe_gemm_a16w4 runs, resolving None the way the op does."""
    if backend is not None:
        return backend
    return "gluon" if get_arch() == "gfx1250" else "triton"


def kernel_variant(K, swizzle_mx_scale, backend=None):
    """Compiled kernel moe_gemm_a16w4 dispatches to for a K-deep projection.

    Re-derived from moe_op_gemm_a16w4.py's own dispatch, not copied from
    moe_gemm_a8w4's variant table -- the two ops do not share the same split:
    a16w4 has no persistent-decode/decode/prefill gluon split on gfx1250 (a
    single `_moe_gemm_a16w4_gluon` kernel covers it); the only backend
    fallback lives on gfx950, where the gluon kernel additionally requires
    K % 256 == 0 and swizzled scales, else the op silently falls back to
    triton.
    """
    if backend_name(backend) != "gluon":
        return "_moe_gemm_a16w4_triton"
    arch = get_arch()
    if arch == "gfx1250":
        return "_moe_gemm_a16w4_gluon"
    elif arch == "gfx950":
        if K % 256 != 0 or swizzle_mx_scale is None:
            return "_moe_gemm_a16w4_triton"  # op's own fallback
        return "_moe_gemm_a16w4_gluon_gfx950_swizzle"
    else:
        return "_moe_gemm_a16w4_triton"


_POOL_FIELDS = [
    "x1",
    "w1",
    "w1_scales",
    "b1",
    "rdata",
    "gather_indx",
    "scatter_indx",
    "swizzle_mx_scale1",
    "x2",
    "w2",
    "w2_scales",
    "b2",
    "swizzle_mx_scale2",
    "y1_bytes",
]


def _generate_pool_inputs(
    batch,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    x_dtype_torch,
    w_dtype,
    TP,
    backend,
):
    """One full weight-init draw: weights, routing, and the layer-1 swiglu
    output layer 2 reads. Regenerated per pool entry, so replaying through the
    pool rotates through distinct weight/routing draws instead of keeping one
    draw's tensors cache-resident -- the role `num_weight_inits` used to play
    via a Python averaging loop in the proton benchmark.
    """
    rank = 0
    dev = f"cuda:{rank}"

    # -- init data --
    # weights
    wg = torch.randn((dim1, n_expts_tot), dtype=torch.float32, device=dev)
    w1 = torch.randn((n_expts_tot, dim1, dim2 // TP), dtype=torch.float32, device=dev)
    w2 = torch.randn(
        (n_expts_tot, dim2 // TP // 2, dim1), dtype=torch.float32, device=dev
    )
    # biases -- always created and passed; the original script has no --bias
    # toggle
    bg = torch.randn((n_expts_tot,), dtype=torch.float32, device=dev)
    b1 = torch.randn((n_expts_tot, dim2 // TP), dtype=torch.float32, device=dev)
    b2 = torch.randn((n_expts_tot, dim1), dtype=torch.float32, device=dev)

    # -- numerics --
    wg, _ = quantize(wg, "bf16")
    w1, w1_scales = quantize(w1, w_dtype)
    w2, w2_scales = quantize(w2, w_dtype)
    w1_scales, swizzle_mx_scale1 = check_and_shuffle_scales(w1_scales, dim2 // TP, dim1)
    w2_scales, swizzle_mx_scale2 = check_and_shuffle_scales(
        w2_scales, dim1, dim2 // TP // 2
    )

    # -- routing + layer-1 activations --
    x1 = torch.randn((batch, dim1), dtype=x_dtype_torch, device=dev)
    logits = gemm_a16w16(x1, wg.T, bg)
    rdata, gather_indx, scatter_indx = routing(logits, n_expts_act)

    def gemm1(out_dtype):
        return moe_gemm_a16w4(
            x1,
            w1,
            None,
            w1_scales,
            None,
            None,
            b1,
            rdata,
            gather_indx=gather_indx,
            swizzle_mx_scale=swizzle_mx_scale1,
            out_dtype=out_dtype,
            apply_swiglu=True,
            backend=backend,
        )

    # layer 2 reads layer 1's swiglu output as-is -- a16w4 never quantizes
    # activations, so there is no quant/calibration step between the two
    # layers (unlike a8w4's static-fp8 hand-off). This call also doubles as
    # the compile warmup.
    y1 = gemm1(x_dtype_torch)
    y1_bytes = y1.numel() * y1.element_size()
    x2 = y1

    return (
        x1,
        w1,
        w1_scales,
        b1,
        rdata,
        gather_indx,
        scatter_indx,
        swizzle_mx_scale1,
        x2,
        w2,
        w2_scales,
        b2,
        swizzle_mx_scale2,
        y1_bytes,
    )


def _layer1_fn(out_dtype, backend):
    def _fn(x1, w1, w1_scales, b1, rdata, gather_indx, swizzle_mx_scale1, **_):
        return moe_gemm_a16w4(
            x1,
            w1,
            None,
            w1_scales,
            None,
            None,
            b1,
            rdata,
            gather_indx=gather_indx,
            swizzle_mx_scale=swizzle_mx_scale1,
            out_dtype=out_dtype,
            apply_swiglu=True,
            backend=backend,
        )

    return _fn


def _layer2_fn(out_dtype, backend):
    def _fn(x2, w2, w2_scales, b2, rdata, scatter_indx, swizzle_mx_scale2, **_):
        return moe_gemm_a16w4(
            x2,
            w2,
            None,
            w2_scales,
            None,
            None,
            b2,
            rdata,
            scatter_indx=scatter_indx,
            swizzle_mx_scale=swizzle_mx_scale2,
            out_dtype=out_dtype,
            apply_swiglu=False,
            backend=backend,
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
    backend,
    layers=LAYERS,
    num_weight_inits=1,
):
    assert dim2 % TP == 0, f"{dim2=}, {TP=}, dim2 must be divisible by TP"
    assert x_dtype in ACT_DTYPES, f"{x_dtype=} must be one of {ACT_DTYPES}"
    assert w_dtype == "mx4", f"a16w4 weights are mxfp4 (E2M1), got {w_dtype}"
    assert layers, "at least one layer must be selected"
    assert set(layers) <= set(LAYERS), f"unknown layer(s) in {layers=}"

    x_dtype_torch = torch.bfloat16 if x_dtype == "bf16" else torch.float16

    _p_gen = functools.partial(
        _generate_pool_inputs,
        batch,
        dim1,
        dim2,
        n_expts_tot,
        n_expts_act,
        x_dtype_torch,
        w_dtype,
        TP,
        backend,
    )
    # pool_len == num_weight_inits: each captured call in the graph replays a
    # distinct weight/routing draw, the same role the old averaging loop played.
    rot_bufs = bencher.build_pool(
        [_p_gen],
        [_POOL_FIELDS],
    )

    layer1_fn = _layer1_fn(x_dtype_torch, backend)
    layer2_fn = _layer2_fn(x_dtype_torch, backend)
    both_fn = _both_fn(layer1_fn, layer2_fn)

    # -- analytic FLOPs / bytes, matching the proton metadata the original
    # bench_moe_gemm_a16w4.py reports: 2*M*N*K per GEMM, and activations +
    # routed-expert weights + bias + matmul output for traffic. mx scales
    # (~1/16 of the weight bytes) are counted; unlike a8w4, there is no
    # activation-side mx8 scale term at all, since a16w4 never quantizes
    # activations. Taken from entry 0: the shapes are identical across entries
    # even though the routed count can drift slightly with the random draw.
    sample = rot_bufs[0]
    n_tokens = sample["gather_indx"].shape[0]  # routed rows == batch * n_expts_act
    routed = int((sample["rdata"].expt_data.hist > 0).sum())  # experts w/ >= 1 token

    def w_bytes(w):
        return (w.numel() * w.element_size() // n_expts_tot) * routed

    def w_scale_bytes(w):
        return (w.numel() * w.element_size() * 2 // 32 // n_expts_tot) * routed

    x1, w1, x2, w2 = sample["x1"], sample["w1"], sample["x2"], sample["w2"]
    b1, b2 = sample["b1"], sample["b2"]

    moe1_flops = 2 * n_tokens * (dim2 // TP) * dim1  # N = dim2 // TP, K = dim1
    moe1_bytes = (
        x1.numel() * x1.element_size()
        + w_bytes(w1)
        + w_scale_bytes(w1)
        + sample["y1_bytes"]
        + (b1.numel() * b1.element_size() // n_expts_tot) * routed
    )

    moe2_flops = 2 * n_tokens * dim1 * (dim2 // TP // 2)  # N = dim1, K = dim2/TP/2
    # y2 is the scatter-compressed [batch, dim1] result; the GEMM writes the
    # uncompressed [n_tokens, dim1] rows the reduction then combines. Computed
    # once here (outside the timed region) purely for its dtype/element size.
    y2 = layer2_fn(**sample)
    moe2_bytes = (
        x2.numel() * x2.element_size()
        + w_bytes(w2)
        + w_scale_bytes(w2)
        + n_tokens * dim1 * y2.element_size()
        + (b2.numel() * b2.element_size() // n_expts_tot) * routed
    )

    # the two projections have different K, so they can land on different
    # backend variants on gfx950 (see kernel_variant's docstring)
    kernel1 = kernel_variant(dim1, sample["swizzle_mx_scale1"], backend)
    kernel2 = kernel_variant(dim2 // TP // 2, sample["swizzle_mx_scale2"], backend)
    kernels = {
        "moe1": kernel1,
        "moe2": kernel2,
        "total": kernel1 if kernel1 == kernel2 else f"{kernel1}+{kernel2}",
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
    TP,
    backend,
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
        f"-E={n_expts_tot}-topk={n_expts_act}-{backend_name(backend)}"
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
        TP,
        backend,  # fixed args
        layers,
        num_weight_inits,
        bench_fn=bench_mlp,  # function to benchmark
        intensity_proxy_name="batch",  # intensity proxy name
        intensity_proxy_values=batch_sizes,  # intensity proxy values to sweep
        out_path=out_csv,  # output path
    )


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="MoE")
    parser.set_defaults(backend="gluon")

    parser.add_argument(
        "--M",
        type=int,
        nargs="+",
        default=[1],
        help="MoE batch sizes M (one or more integers). "
        "If not set, a predermined list of values will be used.",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=(4096, 14336),
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
        default=(8, 2),
        metavar=("TOTAL", "TOPK"),
        help="TOTAL is how many experts the layer holds; TOPK is how many of "
        "them each token is routed to. TOTAL sets the weight tensors' expert "
        "dim, TOPK multiplies the row count each GEMM sees (batch * TOPK "
        "routed rows).",
    )
    parser.add_argument(
        "--act-dtype",
        choices=list(ACT_DTYPES),
        default="bf16",
        help="Activation dtype: bf16 or fp16. Both are unquantized -- "
        "moe_gemm_a16w4 never downcasts activations, only the mxfp4 weights "
        "are quantized. Default: bf16, matching the original "
        "bench_moe_gemm_a16w4.py's hardcoded choice.",
    )
    parser.add_argument(
        "--TP",
        type=int,
        default=1,
        help="Tensor Parallelism(TP)",
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
        "mx4",
        TP=parsed_args.TP,
        backend=parsed_args.backend,
        # dedupe, keeping the canonical report order rather than argv order
        layers=tuple(n for n in LAYERS if n in parsed_args.layers),
        num_weight_inits=parsed_args.num_weight_inits,
        name="moe_gemm_a16w4",
    )


if __name__ == "__main__":
    main()
