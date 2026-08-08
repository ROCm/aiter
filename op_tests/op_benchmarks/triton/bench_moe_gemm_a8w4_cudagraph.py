# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/bench/bench_mlp.py
"""Benchmark the fp8 x mxfp4 MoE MLP (two moe_gemm_a8w4 calls).

Each batch size is measured three times with `triton.testing.do_bench_cudagraph`
and reported as three rows, keyed by the `layer` column:

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

--act-dtype picks how the activations reach the kernel; the weights are mxfp4
either way, which is the w4 half of a8w4:

  fp8   e4m3 with one static scale per tensor, which the kernel multiplies back
        into the accumulator. Layer 1 quantizes its own output in the epilogue
        (Quant_static_scale), so the hand-off to layer 2 costs no separate
        quant kernel -- and so `moe1` here includes that quantization.
  mx8   mxfp8, i.e. e4m3 plus one ue8m0 scale per 32 values along K. Layer 1
        writes bf16 and layer 2's input is quantized separately.

Both static scales are calibrated from the data they quantize (max / 448),
outside the timed region; the layer-2 one needs layer 1's output magnitude, so
it costs one extra untimed bf16 call at setup.

--routed-experts pins how many experts receive tokens -- not to be confused
with the second value of --experts, which is top-k per token. The per-token
expert sets are built so exactly that many are hit whatever the batch size,
which holds the expert weight bytes fixed across a sweep. `batch * top-k`
routed rows cannot reach more experts than there are rows, so a tiny batch
pins fewer; the routed_experts column always reports what routing really used.

Everything except the GEMMs -- gating, routing, and the activation quantization
that feeds each layer -- is built once outside the timed region, so the numbers
cover the two projections and nothing else.

Unlike moe_gemm_a4w4 there is no backend switch: moe_gemm_a8w4 runs the gluon
kernels on gfx1250 and the triton kernel everywhere else. On gfx1250 the
variant follows routing's block_m *and* the projection's K -- persistent decode
for block_m == 16 with K <= 768, plain decode for the rest of block_m == 16,
prefill otherwise -- so the two projections can land on different kernels and
the `kernel` column is per layer. --preshuffle enables the gluon-only gfx1250
WMMA weight preshuffle.
"""

import argparse
import csv
import inspect
from itertools import chain
from pathlib import Path

import torch
import triton

from aiter.ops.shuffle import shuffle_weight_gfx1250
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4
from aiter.ops.triton.moe.moe_routing.routing import _USE_HERD, routing
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp, downcast_to_static_fp8
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.shuffle import shuffle_scale_moe

# measurable layers, in report order; see the module docstring
LAYERS = ("moe1", "moe2", "total")
# activation formats --act-dtype accepts; the weights are always mxfp4
ACT_DTYPES = ("fp8", "mx8")


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
    if get_arch() == "gfx950" and N % 32 == 0 and K % (32 * 8) == 0:
        scale = shuffle_scale_moe(
            scale, arch="gfx950", preshuffle_factor=32, scale_kwidth=8
        )
        return scale, "CDNA4_SCALE"
    elif get_arch() == "gfx1250" and N % 32 == 0 and K % (32 * 8) == 0:
        scale = shuffle_scale_moe(
            scale, arch="gfx1250", preshuffle_factor=32, scale_kwidth=8
        )
        return scale, "GFX1250_SCALE"
    else:
        return scale, None


def preshuffle_weight(w):
    """gfx1250 WMMA weight preshuffle.

    `w` is the mxfp4 weight [E, K // 2, N]; the result is the TDM view
    [E, (K // 2) * 16, N // 16] the gluon kernel reads with PRESHUFFLED=True.
    shuffle_weight_gfx1250() asserts K // 2 % 32 and N % 16.
    """
    return shuffle_weight_gfx1250(w)


def fp8_dtype():
    """e4m3, in the flavour this arch's kernels take."""
    return torch.float8_e4m3fn if get_arch() != "gfx942" else torch.float8_e4m3fnuz


def static_scale_of(x):
    """Per-tensor fp8 scale, as in test_moe_gemm_a8w4.py: `x / scale` fits e4m3
    and the kernel multiplies `scale` back into the accumulator."""
    return x.abs().max().float() / 448.0


def quantize(x, dtype):
    """bf16 / mxfp8 / mxfp4 tensors.

    Static-scale fp8 activations do not come through here: they need
    downcast_to_static_fp8() plus the scale the kernel multiplies back into the
    accumulator, so the setup below quantizes them inline.
    """
    if dtype == "bf16":
        x = x.to(torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)
        return x, None
    elif dtype == "mx8":
        x, scale = downcast_to_mxfp(x, fp8_dtype(), axis=1)
        return x, scale
    else:
        assert dtype == "mx4", f"{dtype=}"
        x, scale = downcast_to_mxfp(x.to(torch.bfloat16), torch.uint8, axis=1)
        return x, scale


def pin_routed_experts(logits, n_routed, n_expts_act):
    """Route to exactly `n_routed` experts -- a pin, not a cap.

    Masking the logits down to a random pool of `n_routed` experts only bounds
    the routed count from above: nothing makes top-k cover the pool, so a small
    batch lands on fewer and the count drifts with the batch size. Choose each
    token's expert set directly instead.

    Every row is rewritten to hold exactly `n_expts_act` finite logits -- the
    token's chosen experts, at their original values, so the gate softmax is
    over real scores -- and -inf everywhere else, the same sentinel `_topk`
    uses for its own out-of-range lanes. Top-k then has no choice but to return
    that set. Each pool expert is claimed by at least one token and the leftover
    slots are filled uniformly at random from the pool, which is the same
    distribution top-k over random logits was already drawing. A histogram with
    zeros in it is the normal case, so hist / block_pid_map stay consistent.

    `batch * n_expts_act` routed rows cannot reach more experts than there are
    rows, so the pool shrinks to fit a tiny batch; the returned count is what
    was actually pinned.
    """
    n_tokens, n_expts_tot = logits.shape
    dev = logits.device
    n_pinned = min(n_routed, n_tokens * n_expts_act)
    pool = torch.randperm(n_expts_tot, device=dev)[:n_pinned]

    # Coverage: pool slot i is claimed by token i % n_tokens. n_pinned <=
    # n_tokens * n_expts_act bounds any one token's claims by n_expts_act, and
    # each slot is claimed once, so no token claims the same expert twice.
    claimed = torch.arange(n_pinned, device=dev) % n_tokens == torch.arange(
        n_tokens, device=dev
    ).unsqueeze(1)

    # Rank the pool per token -- claims first, then a random order over the
    # rest -- and keep the top n_expts_act: every claim survives, and whatever
    # slots are left over come out random.
    rank = torch.rand((n_tokens, n_pinned), device=dev) + claimed
    keep = pool[rank.topk(n_expts_act, dim=-1).indices]

    masked = torch.full_like(logits, float("-inf"))
    masked.scatter_(1, keep, logits.gather(1, keep))
    return masked, n_pinned


def kernel_variant(block_m, K):
    """Compiled kernel moe_gemm_a8w4 dispatches to for a K-deep projection.

    There is no backend argument: the op takes the gluon path iff the arch is
    gfx1250. Within gluon, get_kernel_config_gluon() turns on the persistent
    decode variant for shallow K, so moe1 and moe2 can differ.
    """
    if get_arch() != "gfx1250":
        return "_moe_gemm_a8w4"
    if block_m != 16:
        return "_moe_gemm_a8w4_prefill"
    return "_moe_gemm_a8w4_decode_persistent" if K <= 768 else "_moe_gemm_a8w4_decode"


def bench_mlp_single_weight_init(
    batch,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    x_dtype,
    w_dtype,
    TP,
    preshuffle,
    routed_experts,
    rep,
    layers=LAYERS,
):
    rank = 0
    dev = f"cuda:{rank}"

    assert dim2 % TP == 0, f"{dim2=}, {TP=}, dim2 must be divisible by TP"
    assert x_dtype in ACT_DTYPES, f"{x_dtype=} must be one of {ACT_DTYPES}"
    assert w_dtype == "mx4", f"a8w4 weights are mxfp4 (E2M1), got {w_dtype}"
    if preshuffle:
        assert (
            get_arch() == "gfx1250"
        ), f"--preshuffle needs the gfx1250 gluon kernel, got {get_arch()}"
    if routed_experts is not None:
        # every token needs n_expts_act distinct experts, so the pool can't be smaller
        assert n_expts_act <= routed_experts <= n_expts_tot, (
            f"--routed-experts must be between top-k ({n_expts_act}) and the total "
            f"expert count ({n_expts_tot}), got {routed_experts}"
        )
        # HERD routes top-(k+1) then drops the least batch-popular expert, which
        # shrinks the routed set out from under the pin.
        assert not _USE_HERD, (
            "--routed-experts pins the routed expert set, which HERD routing "
            "undoes; unset AITER_TRITON_USE_HERD"
        )
    assert layers, "at least one layer must be selected"
    assert set(layers) <= set(LAYERS), f"unknown layer(s) in {layers=}"

    # -- init data --
    # weights
    wg = torch.randn((dim1, n_expts_tot), device=dev)
    w1 = torch.randn((n_expts_tot, dim1, dim2 // TP), device=dev)
    w2 = torch.randn((n_expts_tot, dim2 // TP // 2, dim1), device=dev)
    # biases
    bg = torch.randn((n_expts_tot,), device=dev)
    b1 = torch.randn((n_expts_tot, dim2 // TP), device=dev)
    b2 = torch.randn((n_expts_tot, dim1), device=dev)

    # -- numerics --
    wg, _ = quantize(wg, "bf16")
    w1, w1_scale = quantize(w1, w_dtype)
    w2, w2_scale = quantize(w2, w_dtype)
    w1_scale, swizzle_mx_scale1 = check_and_shuffle_scales(w1_scale, dim2 // TP, dim1)
    w2_scale, swizzle_mx_scale2 = check_and_shuffle_scales(
        w2_scale, dim1, dim2 // TP // 2
    )
    if preshuffle:
        w1 = preshuffle_weight(w1)
        w2 = preshuffle_weight(w2)

    # -- routing + layer-1 activations: built once, outside the timed region --
    x = torch.randn((batch, dim1), dtype=torch.bfloat16, device=dev)
    logits = gemm_a16w16(x, wg.T, bg)
    n_pinned = None
    if routed_experts is not None:
        logits, n_pinned = pin_routed_experts(logits, routed_experts, n_expts_act)
    rdata, gather_indx, scatter_indx = routing(logits, n_expts_act)

    static_fp8 = x_dtype == "fp8"
    if static_fp8:
        x1_scale = None
        x1_static_scale = static_scale_of(x)
        x1 = downcast_to_static_fp8(x, x1_static_scale)
    else:
        x1_static_scale = None
        x1, x1_scale = quantize(x, x_dtype)

    def gemm1(out_dtype, quant_static_scale):
        return moe_gemm_a8w4(
            x1,
            w1,
            x1_scale,
            w1_scale,
            x1_static_scale,
            quant_static_scale,
            b1,
            rdata,
            gather_indx=gather_indx,
            swizzle_mx_scale=swizzle_mx_scale1,
            out_dtype=out_dtype,
            apply_swiglu=True,
            preshuffled=preshuffle,
        )

    if static_fp8:
        # Layer 1 quantizes its own swiglu output in the epilogue, so layer 2
        # reads fp8 with no quant kernel in between -- but the scale has to be
        # known up front. Calibrate it from one untimed bf16 call so the
        # hand-off lands in e4m3 range instead of saturating.
        y1_bf16 = gemm1(torch.bfloat16, None)
        x2_static_scale = static_scale_of(y1_bf16)
        del y1_bf16
        out_dtype1, quant_static_scale1 = fp8_dtype(), x2_static_scale
    else:
        x2_static_scale = None
        out_dtype1, quant_static_scale1 = torch.bfloat16, None

    def layer1():
        return gemm1(out_dtype1, quant_static_scale1)

    # layer 2 reads layer 1's swiglu output; get it (and, for mxfp8, its
    # scales) once here so the timed region holds only the two GEMMs. This
    # doubles as the compile warmup.
    y1 = layer1()
    y1_bytes = y1.numel() * y1.element_size()
    if static_fp8:
        # already e4m3, quantized by layer 1 with x2_static_scale
        x2, x2_scale = y1, None
    else:
        x2, x2_scale = quantize(y1, x_dtype)
        del y1

    def layer2():
        return moe_gemm_a8w4(
            x2,
            w2,
            x2_scale,
            w2_scale,
            x2_static_scale,
            None,
            b2,
            rdata,
            scatter_indx=scatter_indx,
            swizzle_mx_scale=swizzle_mx_scale2,
            preshuffled=preshuffle,
        )

    y2 = layer2()
    torch.cuda.synchronize()

    def both():
        layer1()
        layer2()

    # -- analytic FLOPs / bytes, matching the proton metadata the kernel itself
    # reports: 2*M*N*K per GEMM, and activations + routed-expert weights +
    # matmul output for traffic. mx scales (~1/16 of the weight bytes, and of
    # the activation bytes under --act-dtype mx8) and the moe2 scatter
    # reduction are not counted; the reduction's runtime is inside
    # moe_gemm_a8w4 and so is inside the measurement.
    n_tokens = gather_indx.shape[0]  # routed rows == batch * n_expts_act
    routed = int((rdata.expt_data.hist > 0).sum())  # experts that got >= 1 token
    if n_pinned is not None:
        assert (
            routed == n_pinned
        ), f"--routed-experts pinned {n_pinned} experts, routing used {routed}"

    def w_bytes(w):
        return (w.numel() * w.element_size() // n_expts_tot) * routed

    moe1_flops = 2 * n_tokens * (dim2 // TP) * dim1  # N = dim2 // TP, K = dim1
    moe1_bytes = x1.numel() * x1.element_size() + w_bytes(w1) + y1_bytes
    moe2_flops = 2 * n_tokens * dim1 * (dim2 // TP // 2)  # N = dim1, K = dim2/TP/2
    # y2 is the scatter-compressed [batch, dim1] result; the GEMM writes the
    # uncompressed [n_tokens, dim1] rows the reduction then combines.
    moe2_bytes = (
        x2.numel() * x2.element_size()
        + w_bytes(w2)
        + n_tokens * dim1 * y2.element_size()
    )

    # the two projections have the same block_m but different K, so they can
    # land on different gluon variants
    kernel1 = kernel_variant(rdata.block_m, dim1)
    kernel2 = kernel_variant(rdata.block_m, dim2 // TP // 2)
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
        "moe1": (layer1, moe1_flops, moe1_bytes),
        "moe2": (layer2, moe2_flops, moe2_bytes),
        "total": (both, moe1_flops + moe2_flops, moe1_bytes + moe2_bytes),
    }
    measured = {
        name: {
            "latency_ms": triton.testing.do_bench_cudagraph(f, rep=rep),
            "flops": flops,
            "bytes": byts,
            "kernel": kernels[name],
        }
        for name, (f, flops, byts) in to_bench.items()
        if name in layers
    }

    return {
        "layers": measured,
        "block_m": rdata.block_m,
        "routed_experts": routed,
    }


def bench_mlp(
    batch,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    x_dtype,
    w_dtype,
    TP,
    preshuffle,
    routed_experts,
    rep,
    layers=LAYERS,
    num_weight_inits=1,
):
    all_results = []
    for i in range(num_weight_inits):
        result = bench_mlp_single_weight_init(
            batch,
            dim1,
            dim2,
            n_expts_tot,
            n_expts_act,
            x_dtype,
            w_dtype,
            TP,
            preshuffle,
            routed_experts,
            rep,
            layers,
        )
        all_results.append(result)

    num_runs = len(all_results)
    aggregated = {
        "layers": {
            name: {
                **{
                    key: sum(r["layers"][name][key] for r in all_results) / num_runs
                    for key in ("latency_ms", "flops", "bytes")
                },
                # the dispatched kernel depends on batch/topk/E and the layer's
                # K, none of which move with the weight draw
                "kernel": all_results[0]["layers"][name]["kernel"],
            }
            for name in all_results[0]["layers"]
        },
        # routing block_m depends only on batch/topk/E
        "block_m": all_results[0]["block_m"],
        "routed_experts": sum(r["routed_experts"] for r in all_results) / num_runs,
    }

    return aggregated


def roofline_mlp(
    batch_sizes,
    dim1,
    dim2,
    n_expts_tot,
    n_expts_act,
    x_dtype,
    w_dtype,
    TP,
    preshuffle,
    routed_experts,
    rep,
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
    if routed_experts is not None:
        stem += f"-routed={routed_experts}"
    if preshuffle:
        stem += "-preshuffled"
    if tuple(layers) != LAYERS:
        # a partial run holds a subset of the rows, so give it its own file
        stem += "-layers=" + "+".join(layers)
    out_csv = out_dir / f"{stem}.csv"

    compute_roofline(
        dim1,
        dim2,
        n_expts_tot,
        n_expts_act,
        x_dtype,
        w_dtype,
        TP,
        preshuffle,
        routed_experts,
        rep,  # fixed args
        layers,
        num_weight_inits,
        bench_fn=bench_mlp,  # function to benchmark
        intensity_proxy_name="batch",  # intensity proxy name
        intensity_proxy_values=batch_sizes,  # intensity proxy values to sweep
        out_path=out_csv,  # output path
    )


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="Benchmark MoE")

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
        "routed rows). Use --routed-experts to pin how many of the TOTAL "
        "actually receive tokens.",
    )
    parser.add_argument(
        "--act-dtype",
        choices=list(ACT_DTYPES),
        default="fp8",
        help="Activation format: fp8 (e4m3 with one static scale per tensor, "
        "layer 1 emitting layer 2's input directly) or mx8 (mxfp8, one ue8m0 "
        "scale per 32 values along K). Weights are mxfp4 either way. "
        "Default: fp8.",
    )
    parser.add_argument(
        "--preshuffle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Preshuffle the mxfp4 weights for the gfx1250 gluon kernel (default: False).",
    )
    parser.add_argument(
        "--routed-experts",
        type=int,
        default=None,
        help="Pin the number of experts that receive tokens, fixing the "
        "routed_experts column (and so the weight bytes read) across the batch "
        "sweep. Not to be confused with the second value of --experts, which is "
        "top-k per token. batch * top-k routed rows cannot reach more experts "
        "than there are rows, so a batch that small pins fewer. Default: unset, "
        "i.e. random routing over all experts.",
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
        help="do_bench_cudagraph measurement target per batch size, in ms (default: 20).",
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
    quantized_dtypes = [parsed_args.act_dtype, "mx4"]

    roofline_mlp(
        batch_sizes_moe,
        dim1,
        dim2,
        total_experts,
        active_experts,
        quantized_dtypes[0],
        quantized_dtypes[1],
        TP=1,
        preshuffle=parsed_args.preshuffle,
        routed_experts=parsed_args.routed_experts,
        rep=parsed_args.rep,
        # dedupe, keeping the canonical report order rather than argv order
        layers=tuple(n for n in LAYERS if n in parsed_args.layers),
        num_weight_inits=parsed_args.num_weight_inits,
        name="moe_gemm_a8w4",
    )


if __name__ == "__main__":
    main()
