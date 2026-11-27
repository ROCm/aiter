# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/bench/bench_mlp.py
"""Benchmark the int8 (smoothquant) MoE MLP (two moe_gemm_int8_smoothquant calls).

Each batch size is measured with `RotatingBufferBench` (CUDA graph replay) and
reported as up to three rows, keyed by the `layer` column:

  moe1   the gathered up projection alone   (N = dim2 / TP, K = dim1, + silu)
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

Both weights and activations are int8, smoothquant-style: per-output-channel
weight scales (`quantize_weights_int8`) calibrated once per weight-init draw,
and per-token activation scales (`smoothquant_quantize`) that depend on the
*current* input's dynamic range, not a static precomputed value.

This file is the CUDA-graph-bench sibling of bench_moe_gemm_int8_smoothquant.py
(which stays untouched -- its proton-profile-based tight-loop measurement
cannot be ported into RotatingBufferBench's capture/replay model). That
original's `--op-regex .*moe_gemm.*` filter only ever credited the
`moe_gemm_int8_smoothquant` kernels' own flops/bytes -- the gate GEMM,
routing, and both `smoothquant_quantize` calls ran inside its timed loop but
were never part of the measured numbers. This file keeps the same scope:
gating, routing, and *both* smoothquant requantizations (layer 1's input and
layer 2's input) are built once per pool entry, outside the timed region --
not because smoothquant is static (it isn't: it depends on the live input),
but because it was never counted either way, so redoing it per replay would
only cost capture-time work without changing what's measured.

moe_gemm_int8_smoothquant has no `--backend` knob: on gfx942 it silently
switches to a Gluon kernel for a narrow shape band (shallow K, wide N, large
M, no gather/preshuffle -- so only moe2 is ever eligible); every other case
uses the standard Triton kernel. The `kernel` column reports whichever one
that heuristic picks, mirroring the op's own dispatch instead of exposing a
CLI choice that doesn't exist on the op itself.

--preshuffled preshuffles the int8 weights into the MFMA-friendly tile layout
(see `preshuffle_weights`); it is independent of the gfx942 Gluon dispatch
above (preshuffling actually excludes a shape from that path).

Everything except the GEMMs -- gating, routing, and both smoothquant
requantizations -- is built once outside the timed region, so the numbers
cover the two projections and nothing else.
"""

import argparse
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
from aiter.ops.triton.moe.quant_moe import quantize_weights_int8, smoothquant_quantize
from aiter.ops.triton.utils._triton.arch_info import get_arch
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


def kernel_variant(M, N, K, preshuffled, has_gather):
    """Kernel moe_gemm_int8_smoothquant dispatches to, mirroring the op's own
    `use_gluon` heuristic: gfx942, shallow K, wide N, large M, no gather, no
    preshuffle. moe1 always has a gather index, so it never qualifies; moe2
    can land on the Gluon kernel when the shape band matches.
    """
    if (
        get_arch() == "gfx942"
        and K <= 192
        and N >= 1024
        and M >= 4096
        and not preshuffled
        and not has_gather
    ):
        return "_gluon_moe_gemm_int8_smoothquant"
    return "_moe_gemm_int8_smoothquant"


_POOL_FIELDS = [
    "x1",
    "w1",
    "x1_scale",
    "w1_scale",
    "rdata",
    "gather_indx",
    "x2",
    "w2",
    "x2_scale",
    "w2_scale",
    "scatter_indx",
    "y1_bytes",
]


def _generate_pool_inputs(batch, dim1, dim2, n_expts_tot, n_expts_act, TP, preshuffled):
    """One full weight-init draw: weights, routing, and both smoothquant
    requantizations (layer 1's input and, from layer 1's own output, layer
    2's). Regenerated per pool entry, so replaying through the pool rotates
    through distinct weight/routing/activation draws instead of keeping one
    draw's tensors cache-resident -- the role `num_weight_inits` used to play
    via a Python averaging loop. Layer 1 also runs once here (outside the
    timed region), which doubles as the compile warmup.
    """
    rank = 0
    dev = f"cuda:{rank}"

    assert dim2 % TP == 0, f"{dim2=}, {TP=}, dim2 must be divisible by TP"

    # -- init data --
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

    # -- weight quantization (once per weight-init draw) --
    w1_int8, w1_scale = quantize_weights_int8(w1)
    w2_int8, w2_scale = quantize_weights_int8(w2)
    w1_int8 = w1_int8.transpose(1, 2).contiguous().transpose(1, 2)
    w2_int8 = w2_int8.transpose(1, 2).contiguous().transpose(1, 2)
    if preshuffled:
        w1_int8 = preshuffle_weights(w1_int8)
        w2_int8 = preshuffle_weights(w2_int8)

    # -- routing + layer-1 activations --
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
    # layer 2's input requant depends on layer 1's live output range, so it
    # has to happen here, after layer 1 runs -- but like layer 1's own
    # requant, it is still untimed: see the module docstring.
    x2, x2_scale = smoothquant_quantize(y1, fc2_smooth_scale)
    del y1

    return (
        x1,
        w1_int8,
        x1_scale,
        w1_scale,
        rdata,
        gather_indx,
        x2,
        w2_int8,
        x2_scale,
        w2_scale,
        scatter_indx,
        y1_bytes,
    )


def _layer1_fn(preshuffled):
    def _fn(x1, w1, x1_scale, w1_scale, rdata, gather_indx, **_):
        return moe_gemm_int8_smoothquant(
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

    return _fn


def _layer2_fn(preshuffled):
    def _fn(x2, w2, x2_scale, w2_scale, rdata, scatter_indx, **_):
        return moe_gemm_int8_smoothquant(
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
    TP,
    preshuffled,
    layers=LAYERS,
    num_weight_inits=1,
):
    assert dim2 % TP == 0, f"{dim2=}, {TP=}, dim2 must be divisible by TP"
    assert layers, "at least one layer must be selected"
    assert set(layers) <= set(LAYERS), f"unknown layer(s) in {layers=}"

    _p_gen = functools.partial(
        _generate_pool_inputs,
        batch,
        dim1,
        dim2,
        n_expts_tot,
        n_expts_act,
        TP,
        preshuffled,
    )
    # pool_len == num_weight_inits: each captured call in the graph replays a
    # distinct weight/routing/activation draw, the same role the old
    # averaging loop played.
    rot_bufs = bencher.build_pool(
        [_p_gen],
        [_POOL_FIELDS],
    )

    layer1_fn = _layer1_fn(preshuffled)
    layer2_fn = _layer2_fn(preshuffled)
    both_fn = _both_fn(layer1_fn, layer2_fn)

    # -- analytic FLOPs / bytes, matching moe_gemm_int8_smoothquant's own
    # proton launch-metadata exactly (matmul_launch_metadata in
    # _triton_kernels/moe/moe_op_gemm_int8_smoothquant.py): 2*M*N*K flops per
    # GEMM, and read activations + routed-expert weights + matmul output
    # (pre-scatter-reduction row count) for bytes. x_scale/w_scale are fp32
    # per-row/per-channel scalars that the real kernel metadata does not
    # count either, so they are left out here too -- this is not an
    # approximation, it is what was actually measured. Taken from entry 0:
    # shapes are identical across entries even though the routed count can
    # drift slightly with the random draw.
    sample = rot_bufs[0]
    n_tokens = sample["gather_indx"].shape[0]  # routed rows == batch * n_expts_act
    routed = int((sample["rdata"].expt_data.hist > 0).sum())  # experts w/ >= 1 token

    def w_bytes(w):
        return (w.numel() * w.element_size() // n_expts_tot) * routed

    x1, w1, x2, w2 = sample["x1"], sample["w1"], sample["x2"], sample["w2"]
    moe1_flops = 2 * n_tokens * (dim2 // TP) * dim1  # N = dim2 // TP, K = dim1
    moe1_bytes = x1.numel() * x1.element_size() + w_bytes(w1) + sample["y1_bytes"]
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

    # moe1 always gathers, so it never qualifies for the gfx942 Gluon path;
    # moe2 has no gather and can land there if the shape band matches.
    kernel1 = kernel_variant(n_tokens, dim2 // TP, dim1, preshuffled, has_gather=True)
    kernel2 = kernel_variant(
        n_tokens, dim1, dim2 // TP // 2, preshuffled, has_gather=False
    )
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
    TP,
    preshuffled,
    layers=LAYERS,
    num_weight_inits=1,
    name="",
):
    # Put all outputs under logs/<name>/ and write a CSV file (not a directory-as-stem).
    out_dir = Path("logs") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Every knob that changes what is measured goes in the filename, so sweeps
    # over different shapes land side by side instead of overwriting.
    stem = (
        f"int8x-int8w-TP{TP}-dim1={dim1}-dim2={dim2}"
        f"-E={n_expts_tot}-topk={n_expts_act}"
    )
    if preshuffled:
        stem += "-preshuffled"
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
        TP,
        preshuffled,  # fixed args
        layers,
        num_weight_inits,
        bench_fn=bench_mlp,  # function to benchmark
        intensity_proxy_name="batch",  # intensity proxy name
        intensity_proxy_values=batch_sizes,  # intensity proxy values to sweep
        out_path=out_csv,  # output path
    )


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="MoE Int8 SmoothQuant")

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
        "intermediate size, since the silu/swiglu activation halves it. "
        "Together they fix both GEMMs: moe1 is N=DIM2/TP, K=DIM1, and moe2 "
        "is N=DIM1, K=DIM2/TP/2.",
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
        "--preshuffled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Preshuffle the int8 weights into the MFMA-friendly tile layout "
        "(default: False).",
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
        TP=1,
        preshuffled=parsed_args.preshuffled,
        # dedupe, keeping the canonical report order rather than argv order
        layers=tuple(n for n in LAYERS if n in parsed_args.layers),
        num_weight_inits=parsed_args.num_weight_inits,
        name="moe_gemm_int8_smoothquant",
    )


if __name__ == "__main__":
    main()
