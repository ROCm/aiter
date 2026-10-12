# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/bench/bench_mlp.py
"""Benchmark the mxfp4 x mxfp4 MoE MLP (two moe_gemm_a4w4 calls).

Each batch size is measured three times with `triton.testing.do_bench_cudagraph`
and reported as three rows, keyed by the `layer` column:

  moe1   the gathered up projection alone   (N = dim2 / TP, K = dim1, + swiglu)
  moe2   the scattered down projection alone (N = dim1, K = dim2 / TP / 2)
  total  both back to back, i.e. the whole MLP

`total` is not `moe1 + moe2`: an isolated projection replays the same kernel
over and over, so its weights stay cache-resident in a way the real layer does
not. Compare like with like.

--multi-gpu adds a fourth, `combine`, and changes where moe2's rows go: into
mori's real combine staging window on the peer ranks, through the fused
EP_SCATTER epilogue production compiles, instead of a local output buffer. It
needs --ep > 1, that many GPUs, one --M, and torchrun --nproc-per-node=--ep.
The DISPATCH is not simulated -- each rank keeps its own rows and fabricates the
recv layout a dispatch would have produced -- so what it adds over plain --ep is
the peer-memory write traffic and the cross-device barrier, not the all-to-all.

--layers picks which of those are measured, e.g. `--layers moe1 moe2` to skip
the back-to-back run, or `--layers moe1` for one projection on its own. The
setup is unchanged either way (moe2 needs moe1's output to quantize, so layer 1
always runs once outside the timed region), so a row means the same thing
whichever subset it came from.

--routed-experts pins how many experts receive tokens -- not to be confused
with the second value of --experts, which is top-k per token. The per-token
expert sets are built so exactly that many are hit whatever the batch size,
which holds the expert weight bytes fixed across a sweep. `batch * top-k`
routed rows cannot reach more experts than there are rows, so a tiny batch
pins fewer; the routed_experts column always reports what routing really used.

Everything except the GEMMs -- gating, routing, and the activation quantization
that feeds each layer -- is built once outside the timed region, mirroring the
build()/fn() split in mi450-scripts/run_moe_a4w4.py so the numbers are
comparable to that runner (which benches one projection per invocation).

--ep shards the experts the way expert parallelism does: --experts TOTAL is
the count across all ranks, and this process benches ONE rank, holding weights
for TOTAL/--ep of them. Tokens are still routed over all TOTAL experts, so a
rank sees roughly top-k/--ep live gates per token and the rest are dead slots --
which is what sets the tile geometry, and it is not reproducible by shrinking
either the expert count or the batch alone. The DSV4 EP4 decode step is
`--M 2048 --experts 384 6 --ep 4 --routed-experts 384 --balance`: 2048 real
tokens, 12288 gate slots, block_m 32 from 12288/384, a 96-expert histogram with
32 rows each. Default --ep 1, i.e. the plain single-rank `routing()` path,
unchanged.

`moe_gemm_a4w4` defaults to the gluon kernels on gfx1250 --
_moe_gemm_a4w4_decode when routing picks block_m == 16 and _moe_gemm_a4w4_prefill
otherwise -- and the triton kernel elsewhere. --backend pins one instead (gluon
needs gfx1250). --preshuffle enables the gluon-only gfx1250 WMMA weight
preshuffle.
"""

import argparse
import csv
import inspect
import os
from itertools import chain
from pathlib import Path
from types import SimpleNamespace

import torch
import triton

from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import (
    moe_gemm_a4w4,
    mxfp4_quant,
)
from aiter.ops.triton.moe.moe_routing.routing import (
    _USE_HERD,
    ExptData,
    RoutingData,
    _compute_expt_data_internal,
    ep_sort_routing,
    routing,
)
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.shuffle import moe_weight_decode_view, shuffle_scale_moe

# measurable layers, in report order; see the module docstring. "combine" only
# exists under --multi-gpu, where there is a real mori combine to measure.
LAYERS = ("moe1", "moe2", "total")
MG_LAYERS = ("moe1", "moe2", "combine", "total")


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

    # Under --multi-gpu every rank shares one stdout, so nothing is printed
    # where it is produced: each line is gathered and rank 0 emits the whole set
    # in rank order, as ONE write. Printing per rank interleaves mid-line -- the
    # banner alone was three writes -- and no amount of flushing fixes that.
    mori = kwargs.get("mori")

    def emit(line):
        if mori is None:
            print(line, flush=True)
            return
        lines = [None] * mori.world
        mori.dist.all_gather_object(lines, line, group=mori.group)
        if mori.rank == 0:
            print(
                "\n".join(f"[rank {r}] {ln}" for r, ln in enumerate(lines)),
                flush=True,
            )

    # collect performance data
    perfs = []
    bar = "=" * 41
    emit(f"{bar}\n{out_path}...\n{bar}" if mori is None else f"{out_path}")

    for val in intensity_proxy_values:
        perf = inject_proxy_and_call(val, args, kwargs)
        perfs.append((val, perf))

        # one line per value, one "<layer> <us> <TFLOP/s> <TB/s>" group per
        # measurement -- the same three numbers the CSV carries per row
        groups = " | ".join(
            f"{name} {lp['latency_ms'] * 1e3:.2f}us "
            f"{lp['flops'] / lp['latency_ms'] * 1e-9:#.4g} TF/s "
            f"{lp['bytes'] / lp['latency_ms'] * 1e-9:#.4g} TB/s"
            for name, lp in perf["layers"].items()
        )
        emit(
            f"{intensity_proxy_name}: {val:5d} | {groups} | "
            f"{perf['kernel']} block_m={perf['block_m']} "
            f"routed_experts={perf['routed_experts']}"
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
                        "kernel": perf["kernel"],
                        "block_m": perf["block_m"],
                        "routed_experts": perf["routed_experts"],
                    }
                )


def preshuffle_moe_wscale(s):
    """``(E, K//32, N)`` B-scale -> gfx1250 n32k4 layout, same orientation back.

    ``moe_shuffle_scale`` is the n32k4 tile (preshuffle 32, scale kwidth 4) and
    takes the ``(E, N, K//32)`` orientation, so transpose in and back out. Must
    stay in step with ``SCALE_KWIDTH`` in the gfx1250 gluon kernels.
    """
    return moe_shuffle_scale(s.transpose(-1, -2)).transpose(-1, -2)


def check_and_shuffle_scales(scale, N, K):
    if get_arch() == "gfx950" and N % 32 == 0 and K % (32 * 8) == 0:
        scale = shuffle_scale_moe(
            scale, arch="gfx950", preshuffle_factor=32, scale_kwidth=8
        )
        return scale, "CDNA4_SCALE"
    elif get_arch() == "gfx1250" and N % 32 == 0 and K % (32 * 4) == 0:
        # n32k4 layout (scale kwidth 4), so K//32 only needs to divide by 4.
        scale = preshuffle_moe_wscale(scale)
        return scale, "GFX1250_SCALE"
    else:
        return scale, None


def preshuffle_moe_weight(w):
    """gfx1250 WMMA weight preshuffle.

    `w` is the mxfp4 weight [E, K // 2, N]; the result is the TDM view
    [E, (K // 2) * 16, N // 16] the gluon kernel reads with PRESHUFFLE_WEIGHTS=True.
    ``moe_shuffle_weight`` takes the ``(E, N, K // 2)`` orientation and asserts
    K // 2 % 32 and N % 16; ``moe_weight_decode_view`` then reinterprets the
    result (zero-copy) as the flattened view the kernel loads.
    """
    return moe_weight_decode_view(moe_shuffle_weight(w.transpose(-1, -2)))


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


def pin_routed_experts(logits, n_routed, n_expts_act, balance=False):
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

    ``balance`` deals the routes round-robin instead of at random, giving every
    pinned expert the same row count (+-1) and hence completely full block_m
    tiles -- see the comment on the branch below for why that changes timing.

    Returns ``(masked_logits, n_pinned, pool)``. The pool is the expert ids that
    were pinned, which under expert parallelism is the only way to know how many
    of them this rank actually owns.
    """
    n_tokens, n_expts_tot = logits.shape
    dev = logits.device
    n_pinned = min(n_routed, n_tokens * n_expts_act)
    pool = torch.randperm(n_expts_tot, device=dev)[:n_pinned]

    if balance:
        # Round-robin
        flat = torch.arange(n_tokens * n_expts_act, device=dev)
        keep = pool[flat % n_pinned].view(n_tokens, n_expts_act)
    else:
        slot = torch.arange(n_pinned, device=dev)
        score = torch.rand((n_tokens, n_pinned), device=dev)
        score[slot % n_tokens, slot] += 1.0
        keep = pool[score.topk(n_expts_act, dim=-1).indices]

    masked = torch.full_like(logits, float("-inf"))
    masked.scatter_(1, keep, logits.gather(1, keep))
    return masked, n_pinned, pool


def ep_routing(logits, n_expts_act, n_expts_local, rank=0, ep_scatter_geometry=None):
    """
    Routing for ONE rank of an expert-parallel layer, from GLOBAL logits.
    """
    batch, n_expts_tot = logits.shape
    dev = logits.device
    n_gates = batch * n_expts_act

    # Global top-k. The weights only scale moe2's reduction and nothing here is
    # timed, so a softmax over the selected scores is close enough.
    vals, dispatch_ids = torch.topk(logits, n_expts_act, dim=-1)
    dispatch_weights = torch.softmax(vals.float(), dim=-1)
    dispatch_ids = dispatch_ids.to(torch.int32)

    # global id -> local id, -1 for an expert another rank owns. One entry
    # longer than the global count: ep_sort_routing clamps ids into the map and
    # reads the global count back as `numel() - 1`, which is the convention ATOM
    # passes it.
    expert_map = torch.full((n_expts_tot + 1,), -1, dtype=torch.int32, device=dev)
    lo = rank * n_expts_local
    expert_map[lo : lo + n_expts_local] = torch.arange(
        n_expts_local, dtype=torch.int32, device=dev
    )

    # block_m off the GLOBAL expert count -- see the docstring.
    tokens_per_expt = max(1, n_gates // n_expts_tot)
    block_m = max(16, min(triton.next_power_of_2(tokens_per_expt), 128))
    expt_data_bufs = _compute_expt_data_internal(n_expts_local, n_gates, block_m, dev)
    token_offs_raw, token_offs_pad, block_pid_map = expt_data_bufs[:3]

    # No mori staging buffer here, so every row is real -- but still a device
    # tensor, which is the form ep_sort_routing's row mask expects.
    num_local_tokens = torch.tensor([batch], dtype=torch.int32, device=dev)

    hist_full, topk_indx, gate_indx, gate_scal, gate_valid, dst_row = ep_sort_routing(
        dispatch_weights,
        dispatch_ids,
        expert_map,
        n_expts_local,
        num_local_tokens,
        batch,
        n_expts_act,
        n_gates,
        expt_data_bufs,
        ep_scatter_geometry=ep_scatter_geometry,
    )
    # the tail bin holds the sentinel (non-local) count, which gets no tile
    hist = hist_full[:n_expts_local]
    expt_data = ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)
    rdata = RoutingData(block_m, gate_scal, hist, n_expts_local, n_expts_act, expt_data)
    return rdata, topk_indx, gate_indx, gate_valid, dst_row


# ---------------------------------------------------------------------------
# --multi-gpu: GEMM2 delivering into mori's real combine staging window
# ---------------------------------------------------------------------------
def init_mori_combine(rank, world, group, hidden, mtpr, inter, experts, topk):
    """
    The least mori that gives GEMM2 a real place to scatter to.
    """
    import torch.distributed as dist
    from mori.cco import Communicator

    from aiter import ActivationType, QuantType
    from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import MegaMoEGfx1250
    from aiter.ops.flydsl.kernels.mega_moe_gfx1250.types import _from_gpu_ptr
    from aiter.ops.flydsl.moe_common import GateMode

    # ATOM's _cco_per_rank_vmm: every rank could send all its tokens to one peer
    # (world * mtpr recv slots), times 2x headroom for tokens + combine buffers.
    tok_bytes = mtpr * hidden * 2  # bf16 wire
    win_bytes = world * tok_bytes * 2 + (1 << 24)
    per_rank_vmm = 2 * win_bytes + (1 << 28)

    # collective: rank 0 mints the id, everyone joins the same cco fabric
    uid = Communicator.get_unique_id() if rank == 0 else None
    objs = [uid]
    dist.broadcast_object_list(objs, src=0, group=group)
    comm = Communicator.init(world, rank, objs[0], per_rank_vmm=per_rank_vmm)
    comm.barrier()

    mega = MegaMoEGfx1250(
        communicator=comm,
        rank=rank,
        world_size=world,
        model_dim=hidden,
        inter_dim=inter,
        experts=experts,
        topk=topk,
        max_tokens_per_rank=mtpr,
        activation=ActivationType.Silu,
        gate_mode=GateMode.INTERLEAVE.value,
        quant_type=QuantType.per_1x32,
        swiglu_limit=10.0,
    )
    # Read once here, where every rank is aligned and a barrier follows:
    # create_dev_comm() may be COLLECTIVE, so it cannot be done lazily from a
    # per-rank branch. Same contract ATOM's init_mega_transport honours.
    per_rank_size = int(comm.create_dev_comm().per_rank_size)
    comm.barrier()

    cfg = mega._config
    slot_stride = cfg.combine_slot_stride_bytes
    assert per_rank_size % slot_stride == 0, (
        f"per_rank_size={per_rank_size} is not a multiple of the combine slot "
        f"stride {slot_stride}; one row index cannot address both peer and slot"
    )
    peer_rows = per_rank_size // slot_stride
    # local_ptr is this rank's alias of comb_inp; step back to peer 0's. Sized to
    # end at the LAST peer's last slot, not a round world*peer_rows, whose tail
    # would run past the flat space.
    base = mega._arena.local_ptr("comb_inp") - cfg.rank * per_rank_size
    rows = (cfg.world_size - 1) * peer_rows + cfg.max_tokens_per_rank * cfg.topk
    stride_elems = slot_stride // 2  # bf16 wire
    view = _from_gpu_ptr(base, (rows * stride_elems,), torch.bfloat16).as_strided(
        (rows, cfg.hidden_dim), (stride_elems, 1)
    )
    return mega, view, peer_rows


def mori_src_token_map(batch, world, dev):
    return torch.arange(batch, dtype=torch.int32, device=dev)


def do_bench_cudagraph_synced(fn, rep, group):
    import torch.distributed as dist

    with torch.cuda.stream(torch.cuda.Stream()):
        fn()  # warmup
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(5):
            fn()
        end.record()
        torch.cuda.synchronize()
        estimate_ms = start.elapsed_time(end) / 5

        agreed = torch.tensor([estimate_ms], dtype=torch.float64)
        dist.all_reduce(agreed, op=dist.ReduceOp.MAX, group=group)
        estimate_ms = float(agreed[0])
        n_repeat = 1000 if estimate_ms == 0 else max(1, int(rep / estimate_ms))

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_repeat):
                fn()
        torch.cuda.synchronize()

        times = []
        for _ in range(10):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            g.replay()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / n_repeat)
    return sum(times) / len(times)


def backend_name(backend=None):
    """Backend moe_gemm_a4w4 runs, resolving None the way the op does."""
    if backend is not None:
        return backend
    return "gluon" if get_arch() == "gfx1250" else "triton"


def kernel_variant(block_m, backend=None):
    """Compiled kernel moe_gemm_a4w4 dispatches to -- same rule as
    run_moe_a4w4.py's `name` subcommand."""
    if backend_name(backend) != "gluon":
        return "_moe_gemm_a4w4"
    return "_moe_gemm_a4w4_decode" if block_m == 16 else "_moe_gemm_a4w4_prefill"


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
    backend,
    routed_experts,
    rep,
    layers=LAYERS,
    balance=False,
    fused_quant=False,
    ep=1,
    mori=None,
):
    # mori is the --multi-gpu bundle from init_mori_combine (None = single rank)
    rank = 0 if mori is None else mori.rank
    dev = f"cuda:{torch.cuda.current_device()}"

    assert dim2 % TP == 0, f"{dim2=}, {TP=}, dim2 must be divisible by TP"
    # --experts TOTAL is the global count; this process holds one rank's shard.
    assert ep >= 1, f"--ep must be positive, got {ep}"
    assert n_expts_tot % ep == 0, (
        f"--experts TOTAL ({n_expts_tot}) must divide evenly into --ep ({ep}) "
        "equal shards"
    )
    n_expts_local = n_expts_tot // ep
    if mori is not None:
        assert (
            ep == mori.world
        ), f"--multi-gpu needs --ep == world size, {ep} != {mori.world}"
        assert batch == mori.batch, (
            "--multi-gpu sizes the symmetric arena from a single --M; got "
            f"{batch} against the {mori.batch} it was built for"
        )
    assert x_dtype == "mx4", f"FP4 (E2M1) is disabled for x_dtype, got {x_dtype}"
    assert w_dtype == "mx4", f"FP4 (E2M1) is disabled for x_dtype, got {w_dtype}"
    if preshuffle:
        assert (
            get_arch() == "gfx1250"
        ), f"--preshuffle needs the gfx1250 gluon kernel, got {get_arch()}"
    if fused_quant:
        # moe_gemm_a4w4 asserts the gluon backend for out_mx_quant (the triton
        # kernel has no gfx1250 epilogue and would return unwritten scales), and
        # the swiglu halving must leave the packed width 32-aligned.
        assert backend_name(backend) == "gluon", (
            "--fused-quant needs the gluon backend for out_mx_quant, got "
            f"{backend_name(backend)}"
        )
        n_out = (dim2 // TP) // 2  # apply_swiglu=True halves N
        assert n_out % 32 == 0, (
            f"--fused-quant needs the post-swiglu width {n_out} to be a multiple "
            "of 32 (aiter: out_mx_quant requires N_out % 32 == 0)"
        )
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
    _known = MG_LAYERS if mori is not None else LAYERS
    assert set(layers) <= set(_known), f"unknown layer(s) in {layers=}"

    # -- init data --
    if mori is not None:
        # Every rank must agree on which rank owns each gate, or the staging
        # window gets slots written twice and slots never written. Same seed ->
        # same logits -> same routing; the ranks then differ only in which
        # 1/ep slice of the experts they hold. See mori_src_token_map.
        torch.manual_seed(0)
    # The router scores every GLOBAL expert (that is what picks the tile
    # geometry), but the expert weights are only this rank's shard.
    wg = torch.randn((dim1, n_expts_tot), device=dev)
    w1 = torch.randn((n_expts_local, dim1, dim2 // TP), device=dev)
    w2 = torch.randn((n_expts_local, dim2 // TP // 2, dim1), device=dev)
    # biases
    bg = torch.randn((n_expts_tot,), device=dev)
    b1 = torch.randn((n_expts_local, dim2 // TP), device=dev)
    b2 = torch.randn((n_expts_local, dim1), device=dev)

    # -- numerics --
    wg, _ = quantize(wg, "bf16")
    w1, w1_scale = quantize(w1, w_dtype)
    w2, w2_scale = quantize(w2, w_dtype)
    w1_scale, swizzle_mx_scale1 = check_and_shuffle_scales(w1_scale, dim2 // TP, dim1)
    w2_scale, swizzle_mx_scale2 = check_and_shuffle_scales(
        w2_scale, dim1, dim2 // TP // 2
    )
    if preshuffle:
        w1 = preshuffle_moe_weight(w1)
        w2 = preshuffle_moe_weight(w2)

    # -- routing + layer-1 activations: built once, outside the timed region --
    x = torch.randn((batch, dim1), dtype=torch.bfloat16, device=dev)
    logits = gemm_a16w16(x, wg.T, bg)
    n_pinned = pool = None
    if routed_experts is not None:
        logits, n_pinned, pool = pin_routed_experts(
            logits, routed_experts, n_expts_act, balance=balance
        )
    gate_valid = None
    ep_scatter = None
    if ep == 1:
        rdata, gather_indx, scatter_indx = routing(logits, n_expts_act)
    else:
        geometry = None
        if mori is not None:
            from aiter.ops.triton.moe.moe_routing.routing import EpScatterGeometry

            geometry = EpScatterGeometry(
                src_token_map=mori_src_token_map(batch, mori.world, dev),
                max_tokens_per_rank=batch // mori.world,
                peer_rows=mori.peer_rows,
            )
        rdata, gather_indx, scatter_indx, gate_valid, dst_row = ep_routing(
            logits,
            n_expts_act,
            n_expts_local,
            rank=rank,
            ep_scatter_geometry=geometry,
        )
        if mori is not None:
            from aiter.ops.triton.moe.reduce import EpCombineScatter

            # GEMM2 stops reducing locally and places its un-reduced rows into
            # the window instead -- straight from the epilogue (EP_SCATTER_1),
            # which is what production compiles and what a local output buffer
            # cannot show.
            ep_scatter = EpCombineScatter(out=mori.view, dst_row=dst_row)
    x1, x1_scale = mxfp4_quant(x)

    def layer1():
        return moe_gemm_a4w4(
            x1,
            w1,
            x1_scale,
            w1_scale,
            b1,
            rdata,
            gather_indx=gather_indx,
            swizzle_mx_scale=swizzle_mx_scale1,
            preshuffle_weights=preshuffle,
            apply_swiglu=True,
            out_mx_quant=fused_quant,
            backend=backend,
        )

    y1 = layer1()
    if fused_quant:
        x2, x2_scale = y1
        y1_bytes_alloc = (
            x2.numel() * x2.element_size() + x2_scale.numel() * x2_scale.element_size()
        )
    else:
        y1_bytes_alloc = y1.numel() * y1.element_size()
        x2, x2_scale = mxfp4_quant(y1)
    del y1

    def layer2():
        return moe_gemm_a4w4(
            x2,
            w2,
            x2_scale,
            w2_scale,
            b2,
            rdata,
            scatter_indx=scatter_indx,
            # under EP most gate slots belong to other ranks and no GEMM here
            # ever writes them; None (every gate live) is right for --ep 1
            gate_valid=gate_valid,
            ep_scatter=ep_scatter,
            swizzle_mx_scale=swizzle_mx_scale2,
            preshuffle_weights=preshuffle,
            backend=backend,
        )

    y2 = layer2()

    if mori is None:
        combine = None
    else:
        from aiter.ops.flydsl.kernels.mega_moe_gfx1250.mega_moe import Routing

        mega_routing = Routing(
            token_count=batch // mori.world,
            reverse_source_view=mori_src_token_map(batch, mori.world, dev),
        )

        def combine():
            return mori.mega._combine(mega_routing)

        combine()
    torch.cuda.synchronize()

    def both():
        layer1()
        layer2()
        if combine is not None:
            combine()

    # -- analytic FLOPs / bytes, matching run_moe_a4w4.py and the proton metadata
    # the kernel itself reports: 2*M*N*K per GEMM, and activations + routed-expert
    # weights + matmul output for traffic. mx scales (~1/16 of the weight bytes)
    # and the moe2 scatter reduction are not counted; the reduction's runtime is
    # inside moe_gemm_a4w4 and so is inside the measurement.
    n_gates = gather_indx.shape[0]  # every gate slot == batch * n_expts_act
    # Rows this rank's GEMM really computes. Equal to n_gates at --ep 1; under EP
    # the dead slots get no tile, so the activation buffers are sized for n_gates
    # but only this fraction of them is ever touched.
    n_rows = int(rdata.expt_data.hist.sum())
    live_frac = n_rows / n_gates
    routed = int((rdata.expt_data.hist > 0).sum())  # LOCAL experts with >= 1 row
    if n_pinned is not None:
        # under EP the pin is over the global set; only the local slice lands here
        expected = n_pinned if ep == 1 else int((pool < n_expts_local).sum())
        assert routed == expected, (
            f"--routed-experts pinned {n_pinned} experts"
            + (f", {expected} of them local," if ep > 1 else ",")
            + f" routing used {routed}"
        )

    def w_bytes(w):
        return (w.numel() * w.element_size() // n_expts_local) * routed

    y1_bytes = int(y1_bytes_alloc * live_frac)
    moe1_flops = 2 * n_rows * (dim2 // TP) * dim1  # N = dim2 // TP, K = dim1
    moe1_bytes = x1.numel() * x1.element_size() + w_bytes(w1) + y1_bytes
    moe2_flops = 2 * n_rows * dim1 * (dim2 // TP // 2)  # N = dim1, K = dim2/TP/2
    # y2 is the scatter-compressed [batch, dim1] result; the GEMM writes the
    # uncompressed [n_rows, dim1] rows the reduction then combines.
    moe2_bytes = (
        int(x2.numel() * x2.element_size() * live_frac)
        + w_bytes(w2)
        + n_rows * dim1 * y2.element_size()
    )

    # -- benchmark: each projection on its own, then the pair back to back,
    # keeping only what `layers` asked for (in LAYERS order, not argv order).
    # `total` is NOT moe1 + moe2 -- an isolated projection replays one kernel
    # over and over, so its weights stay hotter than they are in the real layer.
    # insertion order is report order, so keep it in MG_LAYERS order
    to_bench = {
        "moe1": (layer1, moe1_flops, moe1_bytes),
        "moe2": (layer2, moe2_flops, moe2_bytes),
    }
    if combine is not None:
        # no matmul in it; the traffic is topk staged slots in, one row out
        mtpr = batch // mori.world
        to_bench["combine"] = (combine, 0, mtpr * (n_expts_act + 1) * dim1 * 2)
    to_bench["total"] = (both, moe1_flops + moe2_flops, moe1_bytes + moe2_bytes)

    measured = {}
    for name, (f, flops, byts) in to_bench.items():
        if name not in layers:
            continue
        if mori is None:
            latency_ms = triton.testing.do_bench_cudagraph(f, rep=rep)
        else:
            # Line the ranks up first: anything holding a cross-device barrier
            # otherwise charges its peers' skew to whoever arrives early, and
            # that lands in the warmup estimate that sets the replay count.
            torch.cuda.synchronize()
            mori.dist.barrier(group=mori.group)
            latency_ms = do_bench_cudagraph_synced(f, rep, mori.group)
        measured[name] = {"latency_ms": latency_ms, "flops": flops, "bytes": byts}

    return {
        "layers": measured,
        "kernel": kernel_variant(rdata.block_m, backend),
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
    backend,
    routed_experts,
    rep,
    layers=LAYERS,
    num_weight_inits=1,
    balance=False,
    fused_quant=False,
    ep=1,
    mori=None,
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
            backend,
            routed_experts,
            rep,
            layers,
            balance=balance,
            fused_quant=fused_quant,
            ep=ep,
            mori=mori,
        )
        all_results.append(result)

    num_runs = len(all_results)
    aggregated = {
        "layers": {
            name: {
                key: sum(r["layers"][name][key] for r in all_results) / num_runs
                for key in ("latency_ms", "flops", "bytes")
            }
            for name in all_results[0]["layers"]
        },
        # routing block_m and the dispatched kernel depend only on batch/topk/E
        "kernel": all_results[0]["kernel"],
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
    backend,
    routed_experts,
    rep,
    layers=LAYERS,
    num_weight_inits=1,
    balance=False,
    fused_quant=False,
    ep=1,
    mori=None,
    name="",
):
    # Put all outputs under logs/<name>/ and write a CSV file (not a directory-as-stem).
    out_dir = Path("logs") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Every knob that changes what is measured goes in the filename, so sweeps
    # over different shapes/backends land side by side instead of overwriting.
    stem = (
        f"{x_dtype}x-{w_dtype}w-TP{TP}-dim1={dim1}-dim2={dim2}"
        f"-E={n_expts_tot}-topk={n_expts_act}"
    )
    if routed_experts is not None:
        stem += f"-routed={routed_experts}"
    if balance:
        stem += "-balanced"
    if fused_quant:
        stem += "-fusedquant"
    if ep > 1:
        stem += f"-ep{ep}"
    if mori is not None:
        # one file per rank: the ranks share a filesystem and their numbers
        # differ (barrier skew is real signal, not noise to average away)
        stem += f"-mori-rank{mori.rank}of{mori.world}"
    stem += f"-{backend_name(backend)}"
    if preshuffle:
        stem += "-preshuffled"
    if tuple(layers) != (MG_LAYERS if mori is not None else LAYERS):
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
        backend,
        routed_experts,
        rep,  # fixed args
        layers,
        num_weight_inits,
        balance=balance,  # forwarded to bench_mlp via compute_roofline's **kwargs
        fused_quant=fused_quant,
        ep=ep,
        mori=mori,
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
        help="TOTAL is how many experts the layer holds ACROSS ALL EP RANKS; "
        "TOPK is how many of them each token is routed to. TOTAL sets the tile "
        "geometry (tokens per expert is batch*TOPK/TOTAL) and, at --ep 1, the "
        "weight tensors' expert dim; with --ep N the weights hold TOTAL/N. TOPK "
        "multiplies the row count each GEMM sees (batch * TOPK gate slots). Use "
        "--routed-experts to pin how many of the TOTAL actually receive tokens.",
    )
    parser.add_argument(
        "--ep",
        type=int,
        default=1,
        help="Expert-parallel world size to simulate. --experts TOTAL is then "
        "the global count and this process benches ONE rank holding TOTAL/EP "
        "experts, while tokens are still routed over all TOTAL -- so a rank sees "
        "~TOPK/EP live gates per token and the rest are dead slots, exactly as "
        "after ATOM's mori all-to-all (minus the transport). This is the only "
        "way to get the production tile geometry: block_m comes from the GLOBAL "
        "expert count and the histogram from the LOCAL one, which no single "
        "routing() call can produce. DSV4 EP4 decode is '--M 2048 --experts 384 "
        "6 --ep 4 --routed-experts 384 --balance'. Default 1 (no sharding).",
    )
    parser.add_argument(
        "--backend",
        choices=["triton", "gluon"],
        default=None,
        help="Kernel backend for moe_gemm_a4w4. Default: unset, i.e. the arch "
        "default (gluon on gfx1250, triton elsewhere). gluon requires gfx1250.",
    )
    parser.add_argument(
        "--preshuffle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Preshuffle the mxfp4 weights for the gfx1250 gluon kernel (default: False).",
    )
    parser.add_argument(
        "--fused-quant",
        action="store_true",
        default=False,
        help="Have GEMM1 emit MXFP4 straight from its epilogue (out_mx_quant, "
        "i.e. HAS_MX_OUT_1) instead of writing bf16 and requantizing in a "
        "separate launch. This is what ATOM runs on gfx1250, and it moves the "
        "requant INSIDE moe1's measurement -- expect moe1 to get slower and the "
        "separate launch to vanish. GEMM2 cannot do it: aiter restricts "
        "out_mx_quant to GEMM1-style calls with no scatter. gluon only.",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        default=False,
        help="Deal the routed experts round-robin so every one receives the same "
        "number of rows, instead of drawing them at random. Fills every block_m "
        "tile, which is what a load-balanced production step does; the random "
        "default leaves ~45%% of experts spilling into a second near-empty tile "
        "(1.45x more tile work on the DSV4 EP decode shape). Also makes the "
        "histogram deterministic run-to-run. Implies --routed-experts (defaults "
        "to all experts).",
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
        "--multi-gpu",
        action="store_true",
        default=False,
        help="Have GEMM2 deliver its rows into mori's real combine staging "
        "window -- PEER memory on the other EP ranks, through the fused "
        "EP_SCATTER epilogue production compiles -- instead of a local output "
        "buffer, and measure the mori combine that drains it as an extra "
        "'combine' row. Needs --ep > 1, at least --ep visible GPUs, exactly one "
        "--M, and torchrun with --nproc-per-node equal to --ep. Only GEMM2 and "
        "the combine change; moe1 is untouched. The dispatch is NOT simulated: "
        "each rank keeps its own rows and fabricates the recv layout a dispatch "
        "would have produced, so what this adds over plain --ep is the peer-"
        "memory write traffic and the cross-device barrier, not the all-to-all.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        choices=MG_LAYERS,
        default=None,
        help="Which layers to measure: moe1 (up projection), moe2 (down "
        "projection), total (both back to back). E.g. '--layers moe1 moe2' to "
        "skip the back-to-back run, or '--layers moe1' for one projection "
        "alone. combine needs --multi-gpu. Default: all of them.",
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
    quantized_dtypes = ["mx4", "mx4"]

    all_layers = MG_LAYERS if parsed_args.multi_gpu else LAYERS
    layers = parsed_args.layers if parsed_args.layers is not None else list(all_layers)
    bad = set(layers) - set(all_layers)
    assert not bad, f"{sorted(bad)} needs --multi-gpu"

    mori = None
    if parsed_args.multi_gpu:
        import torch.distributed as dist

        ep = parsed_args.ep
        assert ep > 1, "--multi-gpu is only meaningful with --ep > 1"
        assert torch.cuda.device_count() >= ep, (
            f"--multi-gpu --ep {ep} needs at least {ep} GPUs, this machine has "
            f"{torch.cuda.device_count()}"
        )
        assert len(batch_sizes_moe) == 1, (
            "--multi-gpu sizes mori's symmetric arena once, so it takes exactly "
            f"one --M; got {batch_sizes_moe}"
        )
        batch = batch_sizes_moe[0]
        assert batch % ep == 0, (
            f"--M {batch} must divide by --ep {ep}: a recv buffer holds "
            "max_tokens_per_rank rows from each of ep peers"
        )
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        assert world == ep, (
            f"--multi-gpu needs one process per EP rank: WORLD_SIZE={world} "
            f"against --ep {ep}. Launch with torchrun --nproc-per-node={ep}."
        )
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world
        )
        group = dist.new_group(backend="gloo")
        mega, view, peer_rows = init_mori_combine(
            rank,
            world,
            group,
            hidden=dim1,
            mtpr=batch // world,
            inter=dim2 // 2,
            experts=total_experts,
            topk=active_experts,
        )
        mori = SimpleNamespace(
            mega=mega,
            view=view,
            peer_rows=peer_rows,
            rank=rank,
            world=world,
            group=group,
            batch=batch,
            dist=dist,
        )

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
        backend=parsed_args.backend,
        routed_experts=(
            parsed_args.routed_experts
            if parsed_args.routed_experts is not None
            # pin_routed_experts is the only thing that deals routes, so a
            # balanced run has to go through it: default the pool to every expert.
            else (total_experts if parsed_args.balance else None)
        ),
        balance=parsed_args.balance,
        fused_quant=parsed_args.fused_quant,
        ep=parsed_args.ep,
        mori=mori,
        rep=parsed_args.rep,
        # dedupe, keeping the canonical report order rather than argv order
        layers=tuple(n for n in all_layers if n in layers),
        num_weight_inits=parsed_args.num_weight_inits,
        name="moe_gemm_a4w4",
    )

    if mori is not None:
        mori.dist.barrier(group=mori.group)
        mega.close()
        mori.dist.destroy_process_group()


if __name__ == "__main__":
    main()
