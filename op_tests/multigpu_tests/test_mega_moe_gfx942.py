# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MegaMoEGfx942 (the gfx942 EP8 a8w4 / a8w8 MegaMoE) accuracy and performance on 8 GPUs.

The layer is DeepSeek-V4-Flash's MoE: H 4096, I 2048, 256 experts (32 per rank),
top-6, silu with swiglu_limit 10. Every output is checked against an fp32 reference
with the clamp. An instance holds either the main path or large_M, so the test builds
one of each; small_M runs on the large_M one (the small kernel for a8w4, the main path
at the small-input capacity for a8w8).
``--quant a8w8`` runs the layer on FP8 128x128 block-scale experts instead of MXFP4
ones; some of their blocks have scale 0 over non-zero bytes (they must contribute
nothing).
``--model`` takes another model's MoE layer over the 8 ranks (``MODELS``):
DeepSeek-V4.1-Flash (H 5120, I 2304, 384 experts, top-6, silu with swiglu_limit 10),
MiMo-V2.6-Flash (H 4096, I 2048, 256 experts, top-8, silu) or Kimi-K3 (routed experts
in its 3584-wide latent space: H 3584, I 3072, 896 experts, top-16, SiTUv2 with beta 4
and linear_beta 25). A layer that ``eligible()`` does not take is skipped, and so is a
path the layer does not run (DeepSeek-V4.1-Flash: large_M).

- test_mega_moe_gfx942: every path (main and large_M, each also on x quantized
  beforehand by aiter's per_1x128 quant ("main pq", "large pq", timed with that quant),
  and small_M) eagerly and as a CUDA graph replay. The ranks hold different token
  counts, one rank holds none, and some rows are padding (every id == experts, as
  sglang marks them, or -1) or have invalid slots.
- test_mega_moe_gfx942_capacity: every rank brings max_tok_per_rank rows. With
  "all_to_one" routing every route of every rank goes to rank 0, which must receive
  all of them (8 x the uniform average; past 2^18 rows at max_tok_per_rank 8192).

The test spawns its 8 processes:

    python3 op_tests/multigpu_tests/test_mega_moe_gfx942.py
"""

import argparse
import itertools
import logging
import os
import socket
import threading
import time

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "8G")

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import (
    shuffle_scale_a16w4,
    shuffle_weight,
    shuffle_weight_a16w4,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils

WORLD = 8
# Each model's MoE layer: H, I, experts, top-k, activation, swiglu_limit (0: no clamp).
MODELS = {
    "DeepSeek-V4-Flash": (4096, 2048, 256, 6, "silu", 10.0),
    "DeepSeek-V4.1-Flash": (5120, 2304, 384, 6, "silu", 10.0),
    "MiMo-V2.6-Flash": (4096, 2048, 256, 8, "silu", 0.0),
    # H is routed_expert_hidden_size: the routed experts run in the latent space.
    "Kimi-K3": (3584, 3072, 896, 16, "situv2", 0.0),
}
SITU_BETA, SITU_LINEAR_BETA = 4.0, 25.0  # Kimi-K3's SiTUv2
H, I, E, TOPK, ACT, SWIGLU_LIMIT = MODELS["DeepSeek-V4-Flash"]
E_LOCAL = E // WORLD
MTPR, SMALL_MTPR = 1024, 128
# Bound on |out - ref| / rms(reference row) over every element of a routed row. The
# kernels quantize x and the gemm1 output to fp8 e4m3 (3 mantissa bits): the largest
# error measured over all cases is 0.31 (8 x 8192 tokens to one rank).
ATOL = 0.5
# A rank whose case makes no progress for this long fails the test: the kernels spin on
# their peers, so a protocol bug hangs every rank instead of raising.
HANG_S = 300
# run_perftest times with CUDA events: under the torch profiler a rank can stall between
# two graph-replayed kernels while its peers spin on it.
PERF = {"num_iters": 20, "num_warmup": 2, "use_cuda_event": True}

# This rank's layer and reference weights (set by _worker; @benchmark logs every
# argument of a test function as a table column, so they are not arguments).
_STATE = {}


def _free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _use_model(name):
    """Point the layer shape (H, I, E, TOPK, ACT, SWIGLU_LIMIT, E_LOCAL) at ``name``'s."""
    global H, I, E, TOPK, ACT, SWIGLU_LIMIT, E_LOCAL
    H, I, E, TOPK, ACT, SWIGLU_LIMIT = MODELS[name]
    E_LOCAL = E // WORLD


def _skip_reason(quant, model="DeepSeek-V4-Flash"):
    gfx = get_gfx_runtime()
    if gfx != "gfx942":
        return f"MegaMoEGfx942 runs on gfx942, this is {gfx}"
    if torch.cuda.device_count() < WORLD:
        return f"needs {WORLD} GPUs, found {torch.cuda.device_count()}"
    try:
        import mori.shmem  # noqa: F401
    except ImportError:
        return "needs mori"
    from aiter.ops.flydsl.kernels.mega_moe_gfx942 import eligible

    shape = {
        "world_size": WORLD,
        "model_dim": H,
        "inter_dim": I,
        "experts": E,
        "topk": TOPK,
    }
    if not eligible(**shape, quant=quant, act=ACT):
        return f"MegaMoEGfx942 ({quant}) does not run {model}'s layer on this device"
    return None


def _make_weights(rank, quant):
    """This rank's experts in the form the layer takes and the fp32 dequantized
    weights for the reference."""
    if quant == "a8w8":
        return _make_weights_a8w8(rank)
    return _make_weights_a8w4(rank)


def _make_weights_a8w8(rank):
    """FP8 128x128 block-scale experts in aiter's fp8 MoE preshuffle
    (``shuffle_weight(w, (16, 16))``, as sglang loads them for aiter). About 3% of the
    blocks have scale 0 over non-zero bytes, which must contribute nothing."""
    gen = torch.Generator(device="cuda").manual_seed(1000 + rank)
    fp8_max = torch.finfo(dtypes.fp8).max

    def block_quant(rows, cols, std):
        shape = (E_LOCAL, rows // 128, 128, cols // 128, 128)
        w = torch.randn(shape, device="cuda", generator=gen).mul_(std)
        zero = torch.rand(
            (E_LOCAL, rows // 128, cols // 128), device="cuda", generator=gen
        ).lt(0.03)
        scale = (w.abs().amax(dim=(2, 4)) / fp8_max).masked_fill(zero, 0.0)
        q = (w / scale.clamp_min(1e-30)[:, :, None, :, None]).to(dtypes.fp8)
        del w
        junk = (torch.rand(shape, device="cuda", generator=gen) * 2 - 1).to(dtypes.fp8)
        q = torch.where(zero[:, :, None, :, None], junk, q)
        ref = q.float() * scale[:, :, None, :, None]
        return q.view(E_LOCAL, rows, cols), scale, ref.view(E_LOCAL, rows, cols)

    w1, w1_scale, w1_ref = block_quant(2 * I, H, H**-0.25)
    w2, w2_scale, w2_ref = block_quant(H, I, I**-0.25)
    kernel = (
        shuffle_weight(w1, layout=(16, 16)).contiguous(),
        w1_scale,
        shuffle_weight(w2, layout=(16, 16)).contiguous(),
        w2_scale,
    )
    return kernel, (w1_ref, w2_ref)


def _make_weights_a8w4(rank):
    """This rank's experts: the standard a16w4 preshuffle (gate_up=False, as sglang
    loads them) and the fp32 dequantized weights for the reference."""
    gen = torch.Generator(device="cuda").manual_seed(1000 + rank)
    quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    w1 = torch.randn(
        (E_LOCAL, 2 * I, H), dtype=dtypes.bf16, device="cuda", generator=gen
    )
    w1_q, w1_scale = quant(w1.mul_(H**-0.25), quant_dtype=dtypes.fp4x2)
    w2 = torch.randn((E_LOCAL, H, I), dtype=dtypes.bf16, device="cuda", generator=gen)
    w2_q, w2_scale = quant(w2.mul_(I**-0.25), quant_dtype=dtypes.fp4x2)
    del w1, w2
    kernel = (
        shuffle_weight_a16w4(w1_q, 16, False),
        shuffle_scale_a16w4(w1_scale, E_LOCAL, False),
        shuffle_weight_a16w4(w2_q, 16, False),
        shuffle_scale_a16w4(w2_scale, E_LOCAL, False),
    )

    def dequant(q, scale, rows, cols):
        values = fp4_utils.mxfp4_to_f32(q).view(E_LOCAL, rows, cols)
        scales = fp4_utils.e8m0_to_f32(scale).view(E_LOCAL, rows, cols // 32)
        return values * scales.repeat_interleave(32, dim=-1)

    ref = (dequant(w1_q, w1_scale, 2 * I, H), dequant(w2_q, w2_scale, H, I))
    return kernel, ref


def _make_inputs(tokens, rank, seed, routing="uniform", padding=True):
    """x bf16 [T, H], route weights f32 [T, k], ids int32 [T, k] of this rank's T rows
    (k = TOPK). ``routing``: "uniform" (top-k of random scores), or "all_to_one" (k of
    rank 0's experts per row). ``padding``: the last row is sglang padding (every id ==
    experts), the one before has no route (-1), and row 0 has an invalid slot of each
    kind."""
    n = tokens[rank]
    gen = torch.Generator(device="cuda").manual_seed(seed * WORLD + rank)
    x = torch.randn((n, H), dtype=dtypes.bf16, device="cuda", generator=gen)
    scores = torch.randn((n, E), device="cuda", generator=gen)
    values, ids = torch.topk(scores, TOPK, dim=-1)
    if routing == "all_to_one":
        order = torch.rand((n, E_LOCAL), device="cuda", generator=gen).argsort(dim=-1)
        ids = order[:, :TOPK]
    ids = ids.to(torch.int32)
    if padding and n >= 3:
        ids[-1] = E
        ids[-2] = -1
        ids[0, 1], ids[0, 4] = -1, E
    return x, values.softmax(dim=-1), ids


def _gather(t, tokens):
    """Every rank's rows of ``t``, concatenated in rank order."""
    buf = t.new_zeros((max(tokens), *t.shape[1:]))
    buf[: t.shape[0]] = t
    parts = [torch.empty_like(buf) for _ in range(WORLD)]
    dist.all_gather(parts, buf)
    return torch.cat([p[:n] for p, n in zip(parts, tokens)])


@torch.no_grad()
def run_torch(x, wts, ids, tokens, rank, ref_weights):
    """fp32 reference: this rank's experts on every rank's rows, summed over ranks.
    Returns this rank's rows and the number of routes of all ranks."""
    x_all, wts_all, ids_all = (_gather(t, tokens) for t in (x, wts, ids))
    w1, w2 = ref_weights
    out = torch.zeros((x_all.shape[0], H), dtype=torch.float32, device=x.device)
    for e in range(E_LOCAL):
        rows, slots = torch.nonzero(ids_all == rank * E_LOCAL + e, as_tuple=True)
        if not rows.numel():
            continue
        inp = x_all[rows].float()
        gate, up = inp @ w1[e, :I].T, inp @ w1[e, I:].T
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        if ACT == "situv2":
            act = SITU_BETA * torch.tanh(gate / SITU_BETA) * torch.sigmoid(gate)
            act = act * SITU_LINEAR_BETA * torch.tanh(up / SITU_LINEAR_BETA)
        else:
            act = F.silu(gate) * up
        down = act @ w2[e].T
        out.index_add_(0, rows, down * wts_all[rows, slots, None])
    dist.all_reduce(out)
    start = sum(tokens[:rank])
    routes = int(((ids_all >= 0) & (ids_all < E)).sum())
    return out[start : start + x.shape[0]], routes


def _all_max(value):
    t = torch.tensor([float(value)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t)


def _check(name, out, ref, routed):
    """-> (mismatch fraction, max |out - ref| / rms(reference row)), worst rank."""
    out = out.float()
    rms = ref[routed].pow(2).mean(dim=-1, keepdim=True).sqrt()
    err = checkAllclose(
        ref[routed] / rms,
        out[routed] / rms,
        rtol=0,
        atol=ATOL,
        msg=f"{name} routed rows: ",
    )
    # Rows without a route must come out as zeros, not as stale values.
    err += checkAllclose(
        torch.zeros_like(out[~routed]),
        out[~routed],
        rtol=0,
        atol=0,
        msg=f"{name} rows without a route: ",
    )
    worst = ((out[routed] - ref[routed]) / rms).abs().max() if routed.any() else 0.0
    return _all_max(err), _all_max(worst)


def _perf(tokens, rank, routes):
    """(FLOPs, bytes) per GPU: every route's two GEMMs; the local experts' weights
    (fp4 + e8m0 scales, or fp8 + fp32 128x128 scales), this rank's bf16 rows in and out,
    and the fp8 rows it receives."""
    flops = 2 * routes * 3 * H * I / WORLD
    weight_bytes = 1 + 4 / 128**2 if _STATE["quant"] == "a8w8" else 1 / 2 + 1 / 32
    nbytes = (
        E_LOCAL * 3 * H * I * weight_bytes
        + 2 * tokens[rank] * H * 2
        + routes / WORLD * H
    )
    return flops, nbytes


def _prequant(x):
    """``forward_prequant``'s input: x quantized by aiter's per_1x128 fp8 quant."""
    if not x.shape[0]:
        return x.new_empty(x.shape, dtype=dtypes.fp8), x.new_empty(
            (0, H // 128), dtype=torch.float32
        )
    return aiter.get_torch_quant(aiter.QuantType.per_1x128)(x, quant_dtype=dtypes.fp8)


def _run_paths(tokens, paths, inputs, stale, graph, ret):
    """Run each path on ``inputs`` (after one call on ``stale`` inputs, so a row the call
    does not write keeps a stale value) and check it; fills ``ret``."""
    rank, layers = _STATE["rank"], _STATE["layers"]
    x, wts, ids = inputs
    ref, routes = run_torch(x, wts, ids, tokens, rank, _STATE["ref_weights"])
    routed = ((ids >= 0) & (ids < E)).any(dim=-1)
    flops, nbytes = _perf(tokens, rank, routes)
    for path in paths:
        # "main" / "main pq" on the main path's instance, the others on large_M's.
        mega = layers[path.split()[0] if path != "small" else "large"]

        def call(x, wts, ids, path=path, mega=mega):
            if path == "small":
                return mega.forward_small_m(x, wts, ids, max_tokens=max(tokens))
            if path.endswith(" pq"):
                return mega.forward_prequant(*_prequant(x), wts, ids)
            return mega.forward(x, wts, ids)

        cuda_graph = None
        if graph:
            static = [t.clone() for t in stale]
            call(*static)
            torch.cuda.synchronize()
            cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph):
                out = call(*static)
            cuda_graph.replay()
            for dst, src in zip(static, inputs):
                dst.copy_(src)
            _, us = run_perftest(cuda_graph.replay, **PERF)
        else:
            call(*stale)
            out, us = run_perftest(call, *inputs, **PERF)
        torch.cuda.synchronize()
        _STATE["beat"] = time.monotonic()
        name = f"{path} {'graph' if graph else 'eager'} {tokens}"
        err, worst = _check(name, out, ref, routed)
        us = _all_max(us)
        ret[f"{path} us"] = us
        ret[f"{path} TFLOPS"] = flops / us / 1e6
        ret[f"{path} TB/s"] = nbytes / us / 1e6
        ret[f"{path} err"] = err
        ret[f"{path} max_err"] = worst
        del cuda_graph
    for path in paths:
        assert ret[f"{path} err"] == 0, (
            f"{path} {tokens}: {ret[f'{path} err']:.2%} of elements outside the "
            f"tolerance, max |out - ref| / rms {ret[f'{path} max_err']:.3f} "
            f"(atol {_STATE['atol']})"
        )
    return ret


@benchmark()
def test_mega_moe_gfx942(tokens, graph):
    """``tokens``: rows per rank (the same tuple on every rank); ``graph``: replay a
    captured CUDA graph instead of eager calls."""
    rank, layers = _STATE["rank"], _STATE["layers"]
    seed = _STATE["case"] = _STATE.get("case", 0) + 2
    inputs = _make_inputs(tokens, rank, seed)
    stale = _make_inputs(tokens, rank, seed + 1, padding=False)
    paths = ["main", "main pq"]
    if layers["large"].large_m_available:
        paths += ["large", "large pq"]
    # small_M takes decode batches only.
    if max(tokens) <= layers["large"].small_m_mtpr:
        paths.append("small")
    return _run_paths(tokens, paths, inputs, stale, graph, {"gfx": get_gfx_runtime()})


@benchmark()
def test_mega_moe_gfx942_capacity(mtpr, routing):
    """Every rank brings ``mtpr`` (the layer's max_tok_per_rank) rows; ``routing``:
    "uniform" or "all_to_one" (every route of every rank to rank 0)."""
    rank = _STATE["rank"]
    tokens = (mtpr,) * WORLD
    seed = _STATE["case"] = _STATE.get("case", 0) + 2
    inputs = _make_inputs(tokens, rank, seed, routing, padding=False)
    stale = _make_inputs(tokens, rank, seed + 1, padding=False)
    paths = ["main"] + (
        ["large"] if _STATE["layers"]["large"].large_m_available else []
    )
    return _run_paths(tokens, paths, inputs, stale, False, {"gfx": get_gfx_runtime()})


def _layer(rank, weights, mtpr, small, large):
    from aiter.ops.flydsl.kernels.mega_moe_gfx942 import MegaMoEGfx942

    w1, w1_scale, w2, w2_scale = weights
    mega = MegaMoEGfx942(
        rank=rank,
        world_size=WORLD,
        model_dim=H,
        inter_dim=I,
        experts=E,
        topk=TOPK,
        quant=_STATE["quant"],
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
        max_tok_per_rank=mtpr,
        act=ACT,
        swiglu_limit=SWIGLU_LIMIT,
        small_m_mtprs=(SMALL_MTPR,),
        small_m_enabled=small,
        large_m_enabled=large,
    )
    mega.bind_weights(w1, w1_scale, w2, w2_scale)
    return mega


def _summary(title, rows, rank):
    if rank == 0 and rows:
        aiter.logger.info(
            "%s summary (markdown):\n%s",
            title,
            pd.DataFrame(rows).to_markdown(index=False),
        )


def _watchdog(rank):
    while True:
        time.sleep(10)
        if time.monotonic() - _STATE["beat"] > HANG_S:
            print(
                f"rank {rank}: no progress for {HANG_S} s, a kernel hangs", flush=True
            )
            os._exit(1)


def _worker(rank, port, args):
    _use_model(args.model)
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", rank=rank, world_size=WORLD, device_id=device
    )
    import mori.shmem as ms

    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    if rank:
        aiter.logger.setLevel(logging.WARNING)
    _STATE["beat"] = time.monotonic()
    threading.Thread(target=_watchdog, args=(rank,), daemon=True).start()
    try:
        from aiter.ops.flydsl.kernels.mega_moe_gfx942 import MegaMoEGfx942

        quant = _STATE["quant"] = args.quant
        weights, ref_weights = _make_weights(rank, quant)
        weights = MegaMoEGfx942.prepare_weights(
            *weights, w1_gate_up_interleaved=False, quant=quant
        )
        _STATE.update(rank=rank, ref_weights=ref_weights)
        _STATE["layers"] = {
            "large": _layer(rank, weights, MTPR, small=True, large=True),
            "main": _layer(rank, weights, MTPR, small=False, large=False),
        }
        rows = [
            test_mega_moe_gfx942(tokens, graph)
            for tokens, graph in itertools.product(args.tokens, args.graph)
        ]
        title = f"MegaMoEGfx942 {quant} {args.model}"
        _summary(f"{title}, max_tok_per_rank {MTPR}", rows, rank)
        rows = []
        for mtpr in args.mtpr:
            _STATE["layers"] = {
                "large": _layer(rank, weights, mtpr, small=False, large=True),
                "main": _layer(rank, weights, mtpr, small=False, large=False),
            }
            rows += [
                test_mega_moe_gfx942_capacity(mtpr, routing) for routing in args.routing
            ]
        _summary(f"{title} at full capacity", rows, rank)
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        ms.shmem_finalize()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="MegaMoEGfx942 on 8 GPUs",
    )
    parser.add_argument(
        "-q",
        "--quant",
        choices=["a8w4", "a8w8"],
        default="a8w4",
        help="the layer's quant: a8w4 (MXFP4 experts) or a8w8 (FP8 128x128 "
        "block-scale experts)",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS),
        default="DeepSeek-V4-Flash",
        help="whose MoE layer, split over the 8 ranks",
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(37, 1, 128, 5, 0, 100, 7, 77), (300, 1, 37, 999, 0, 128, 700, 64)],
        help=f"rows of each of the {WORLD} ranks (at most {MTPR}), "
        "e.g. -t 37,1,128,5,0,100,7,77",
    )
    parser.add_argument(
        "-g",
        "--graph",
        type=dtypes.str2bool,
        nargs="*",
        default=[False, True],
        help="replay a captured CUDA graph (1) or call eagerly (0)",
    )
    parser.add_argument(
        "-m",
        "--mtpr",
        type=int,
        nargs="*",
        default=None,
        help="max_tok_per_rank of the full-capacity layers (default: 2048 and the "
        "largest, 8192 at top-6, 4096 at top-8)",
    )
    parser.add_argument(
        "-r",
        "--routing",
        nargs="*",
        choices=["uniform", "all_to_one"],
        default=["uniform", "all_to_one"],
        help="routing of the full-capacity calls",
    )
    args = parser.parse_args()
    _use_model(args.model)
    # The worst-case receive buffer (8 x max_tok_per_rank x top-k rows) stays below 2 GiB.
    args.mtpr = args.mtpr or [2048, 8192 if TOPK <= 6 else 4096]
    reason = _skip_reason(args.quant, args.model)
    if reason:
        aiter.logger.warning("test_mega_moe_gfx942 skipped: %s", reason)
        return
    for tokens in args.tokens:
        if not isinstance(tokens, tuple) or len(tokens) != WORLD or max(tokens) > MTPR:
            parser.error(f"--tokens takes {WORLD} counts <= {MTPR}, got {tokens}")
    mp.spawn(_worker, args=(_free_port(), args), nprocs=WORLD, join=True)


if __name__ == "__main__":
    main()
