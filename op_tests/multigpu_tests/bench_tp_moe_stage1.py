# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Stage-by-stage comparison of the fused TP MoE Stage1 against the NCCL baseline.

Two arms run in the same process, over the same weights, alternating per size so
that machine state is shared:

* the reference arm is ``TPMoEStage1NCCLRef``, the phase-1 pipeline:
  ``NCCL all-gather x3 -> moe_sorting -> quantize -> GEMM1``, four launches;
* the fused arm is ``TPMoEStage1``, which quantizes locally, ships only the
  packed routing metadata over one collective, sorts, and then pushes the
  activations over P2P and runs GEMM1 inside a single kernel launch.

Both arms are broken into four stages so that a regression can be attributed to
one of them. The reference arm additionally reports its prequantized entry
(``forward_prequant``), which gathers FP8+scale instead of BF16 -- that column
is what told phase 1 the gather was latency-bound rather than bandwidth-bound.

Timing: ``dist.barrier()`` before every timed iteration, CUDA events around each
stage, median over ``--iters``, then ``all_reduce(MAX)`` across ranks because the
slowest rank sets the end-to-end cost.

Run with:

    PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
        op_tests/multigpu_tests/bench_tp_moe_stage1.py --m-local 1,8,64,128,256
"""

import argparse
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

# FlyDSL's disk-cache key walks *functions* reachable from the launcher and never
# reads the source of a class, so editing any loader class leaves the key
# unchanged and a stale binary gets reused. Fingerprinting the whole mega_moe
# package restores the dependency. Must precede the flydsl import.
_MEGA_MOE_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "..", "aiter", "ops", "flydsl", "kernels", "mega_moe")
)
_extra = os.environ.get("FLYDSL_EXTRA_SOURCE_DIRS", "")
os.environ["FLYDSL_EXTRA_SOURCE_DIRS"] = (
    f"{_extra}:{_MEGA_MOE_DIR}" if _extra else _MEGA_MOE_DIR
)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

sys.path.insert(0, _HERE)

from aiter.ops.flydsl.kernels.mega_moe.tp_moe_stage1 import TPMoEStage1  # noqa: E402
from aiter.ops.quant import fused_dynamic_mxfp8_quant_moe_sort  # noqa: E402
from aiter.utility.fp4_utils import moe_mxfp4_sort  # noqa: E402
from tp_moe_stage1_nccl_ref import TPMoEStage1NCCLRef  # noqa: E402
from tp_moe_stage1_ref import build_mxfp4_w1  # noqa: E402

NETWORK = dict(model_dim=7168, experts=384, topk=6, swiglu_limit=10.0)
STAGE1_KERNEL = "flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_xcd4_kw4_fp8"


def _setup_dist():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("cpu:gloo,cuda:nccl", device_id=device)
    return rank, world, device


def _random_routes(m, experts, topk, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack(
        [torch.randperm(experts, generator=g)[:topk] for _ in range(m)]
    ).to(device=device, dtype=torch.int32)
    w = torch.rand((m, topk), generator=g).to(device=device, dtype=torch.float32)
    return ids, w / w.sum(dim=-1, keepdim=True)


def _timed(steps):
    """Run ``steps`` in order. Returns per-stage device ms, then host-enqueue ms.

    The host column is the wall time the CPU spent *submitting* the stage. When
    it matches the device column, the GPU had nothing queued and the stage was
    limited by dispatch rather than by work -- which is exactly the failure mode
    a fused path with very little upstream GPU work can fall into.
    """
    n = len(steps)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(n + 1)]
    host = []
    ev[0].record()
    t0 = time.perf_counter()
    for i, fn in enumerate(steps):
        fn()
        ev[i + 1].record()
        t1 = time.perf_counter()
        host.append((t1 - t0) * 1e3)
        t0 = t1
    torch.cuda.synchronize()
    return [ev[i].elapsed_time(ev[i + 1]) for i in range(n)] + host


def _stages_bf16(op, x, wts, ids, m_global):
    """Replay TPMoEStage1NCCLRef.forward with events between stages."""
    s = {}

    def gather():
        s["x_g"], s["wts_g"], s["ids_g"] = op._all_gather_inputs(x, wts, ids)

    def sort():
        s["sids"], _, s["seids"], s["nvalid"] = op._sort(s["ids_g"], s["wts_g"])

    def quant():
        s["a"], s["a_scale"] = fused_dynamic_mxfp8_quant_moe_sort(
            s["x_g"],
            sorted_ids=s["sids"],
            num_valid_ids=s["nvalid"],
            token_num=m_global,
            topk=op.topk,
            block_size=op.sort_block_m,
            sorted_weights=None,
        )

    def gemm1():
        op._run_gemm1(s["a"], s["a_scale"], s["sids"], s["seids"], s["nvalid"])

    return _timed([gather, sort, quant, gemm1])


def _stages_prequant(op, x_q, x_scale, wts, ids, m_global):
    """Replay TPMoEStage1NCCLRef.forward_prequant with events between stages."""
    s = {}

    def gather():
        s["x_g"], s["wts_g"], s["ids_g"] = op._all_gather_inputs(x_q, wts, ids)
        s["scale_g"] = op._all_gather_one(x_scale)

    def sort():
        s["sids"], _, s["seids"], s["nvalid"] = op._sort(s["ids_g"], s["wts_g"])

    def scale_sort():
        s["a_scale"] = moe_mxfp4_sort(
            s["scale_g"].view(m_global, 1, -1),
            s["sids"],
            s["nvalid"],
            m_global,
            op.sort_block_m,
        )

    def gemm1():
        op._run_gemm1(s["x_g"], s["a_scale"], s["sids"], s["seids"], s["nvalid"])

    return _timed([gather, sort, scale_sort, gemm1])


def _stages_fused(op, x, wts, ids):
    """Replay TPMoEStage1.forward with events between stages.

    Mirrors the operator's own forward() step for step; kept here rather than
    instrumented in-place so the production path carries no timing code.
    """
    m_local = int(x.shape[0])
    s = {}

    def quantize():
        s["x_q"], s["x_scale"] = op.quantize(x)

    def metadata():
        meta = torch.empty((m_local, 2 * op.topk), dtype=torch.int32, device=op.device)
        meta[:, : op.topk] = ids
        meta[:, op.topk :] = wts.view(torch.int32)
        meta_g = op._all_gather_one(meta)
        s["ids_g"] = meta_g[:, : op.topk].contiguous()
        s["wts_g"] = meta_g[:, op.topk :].contiguous().view(torch.float32)

    def sort():
        s["sids"], _, s["seids"], s["nvalid"] = op._sort(s["ids_g"], s["wts_g"])

    def fused():
        op._run_fused(
            s["x_q"], s["x_scale"], s["sids"], s["seids"], s["nvalid"], m_local
        )

    return _timed([quantize, metadata, sort, fused])


def _median_stages(fn, iters, warmup):
    """Barrier before each timed iteration so the reading excludes rank skew."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        dist.barrier()
        torch.cuda.synchronize()
        samples.append(fn())
    return [statistics.median(s[i] for s in samples) for i in range(len(samples[0]))]


def _max_across_ranks(values, device):
    """The slowest rank sets the end-to-end cost, so reduce with MAX."""
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.tolist()


def _run(args, rank, world, device):
    model_dim = NETWORK["model_dim"]
    experts, topk = NETWORK["experts"], NETWORK["topk"]
    inter_dim = args.inter_dim
    sizes = [int(v) for v in args.m_local.split(",")]

    _, _, w1_shuf, w1_scale_shuf = build_mxfp4_w1(
        experts, inter_dim, model_dim, device, seed=2026
    )
    common = dict(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        w1=w1_shuf,
        w1_scale=w1_scale_shuf,
        device=device,
        swiglu_limit=NETWORK["swiglu_limit"],
        stage1_kernel_name=STAGE1_KERNEL,
    )
    ref = TPMoEStage1NCCLRef(**common)
    # The symmetric receive buffers are a collective allocation sized once, so
    # they must cover the largest size in the sweep.
    fused = TPMoEStage1(max_tok_per_rank=max(sizes), **common)

    bf16_row = model_dim * 2
    fp8_row = model_dim + model_dim // 32

    if rank == 0:
        print(
            f"world={world} model_dim={model_dim} inter_dim={inter_dim} "
            f"experts={experts} topk={topk} iters={args.iters} "
            f"max_tok_per_rank={max(sizes)}"
        )
        print(f"gather row bytes: bf16={bf16_row}  fp8+scale={fp8_row}")
        print()

    for m_local in sizes:
        gx = torch.Generator(device="cpu").manual_seed(7000 + rank * 31 + m_local)
        x = torch.randn((m_local, model_dim), generator=gx).to(
            device=device, dtype=torch.bfloat16
        ) * (model_dim**-0.25)
        ids, wts = _random_routes(m_local, experts, topk, device, seed=99 + rank)
        x_q, x_scale = ref.quantize(x)
        m_global = ref.m_logical_for(m_local)

        b = _median_stages(
            lambda: _stages_bf16(ref, x, wts, ids, m_global), args.iters, args.warmup
        )
        p = _median_stages(
            lambda: _stages_prequant(ref, x_q, x_scale, wts, ids, m_global),
            args.iters,
            args.warmup,
        )
        f = _median_stages(
            lambda: _stages_fused(fused, x, wts, ids), args.iters, args.warmup
        )
        merged = _max_across_ranks(b + p + f, device)
        b, p, f = merged[:8], merged[8:16], merged[16:]

        if rank != 0:
            continue

        tot_b, tot_p, tot_f = sum(b[:4]), sum(p[:4]), sum(f[:4])
        rx_bf16 = (world - 1) * m_local * bf16_row
        bw_bf16 = rx_bf16 / (b[0] * 1e-3) / 1e9

        print(f"===== m_local={m_local}  m_global={m_global} =====")
        print("  (ms, median over iters, max over ranks; host = CPU submit time)")
        print("  NCCL reference")
        print(
            f"  {'stage':<16}{'bf16 dev':>12}{'bf16 host':>12}"
            f"{'fp8 dev':>12}{'fp8 host':>12}"
        )
        ref_names = ["all-gather", "moe_sorting", "quant/scale", "gemm1"]
        for i, name in enumerate(ref_names):
            print(
                f"  {name:<16}{b[i]:>12.4f}{b[i + 4]:>12.4f}"
                f"{p[i]:>12.4f}{p[i + 4]:>12.4f}"
            )
        print(
            f"  {'TOTAL':<16}{tot_b:>12.4f}{sum(b[4:]):>12.4f}"
            f"{tot_p:>12.4f}{sum(p[4:]):>12.4f}"
        )
        print("  fused path")
        print(f"  {'stage':<16}{'dev':>12}{'host':>12}")
        fused_names = ["local quantize", "metadata coll", "moe_sorting", "fused kernel"]
        for i, name in enumerate(fused_names):
            print(f"  {name:<16}{f[i]:>12.4f}{f[i + 4]:>12.4f}")
        print(f"  {'TOTAL':<16}{tot_f:>12.4f}{sum(f[4:]):>12.4f}")
        print(
            f"  speedup vs bf16 reference: {tot_b / tot_f:.3f}x "
            f"({100 * (tot_b - tot_f) / tot_b:+.1f}% time)"
        )
        print(
            f"  gather rx/rank: bf16 {rx_bf16/1e6:.2f} MB @ {bw_bf16:.1f} GB/s | "
            f"fused kernel absorbs the push"
        )
        print(
            f"  reference overhead outside gemm1: {tot_b - b[3]:.4f} ms  |  "
            f"fused overhead outside the fused kernel: {tot_f - f[3]:.4f} ms"
        )
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-local", default="1,8,64,128,256")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--inter-dim", type=int, default=384)
    args = ap.parse_args()

    import mori.shmem as ms
    import torch._C._distributed_c10d as c10d

    rank, world, device = _setup_dist()
    # The fused path's symmetric buffers live in Mori SHMEM, which needs the TP
    # group registered under a name before it can be initialized.
    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    try:
        _run(args, rank, world, device)
        dist.barrier()
    finally:
        try:
            ms.shmem_finalize()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
