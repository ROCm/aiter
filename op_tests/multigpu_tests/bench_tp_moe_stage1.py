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
import inspect
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

from aiter.ops.flydsl.kernels.mega_moe.tp_fused_stage1 import (  # noqa: E402
    compile_tp_fused_stage1,
)
from aiter.ops.flydsl.kernels.mega_moe.tp_gemm1 import run_tp_gemm1  # noqa: E402
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
        # m_local is what tells the operator to quantize straight into the
        # symmetric source slab under pull. Dropping it here would leave the
        # rows in an ordinary tensor and push a staging copy into the fused
        # stage, which is not what forward() does.
        s["x_q"], s["x_scale"] = op.quantize(x, m_local=m_local)

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


# --------------------------------------------------------------------------
# Controlled arm: the SAME GEMM1 kernel, run either after a separate push
# (two launches) or inside the fused kernel (one launch). Everything except
# the launch boundary is forced equal and asserted below.
# --------------------------------------------------------------------------

# The knobs both compile paths accept. Values are the compile-time defaults of
# compile_tp_fused_stage1, i.e. what TPMoEStage1 actually runs in production;
# _assert_same_config() re-derives them from the signature and fails if that
# ever stops being true.
CTRL_CFG = dict(
    tile_n=256,
    tile_k=256,
    num_waves=4,
    num_cu=256,
    grid_mult=4,
    swizzle_a=True,
    pipe_weights=True,
    mfma_amajor=False,
    async_a_copy=False,
    waves_per_eu_hint=2,
)


def _assert_same_config(runner, sort_block_m, swiglu_limit):
    """Force Arm A's GEMM1 to compile with Arm B's exact parameters.

    Returns the kwargs to hand ``run_tp_gemm1``. Raises if any parameter the
    fused kernel uses cannot be reproduced on the standalone path.
    """
    from aiter.ops.flydsl.kernels.mega_moe import tp_gemm1 as _tp_gemm1_mod

    g = runner.gather
    fused_sig = inspect.signature(compile_tp_fused_stage1).parameters
    gemm_sig = inspect.signature(_tp_gemm1_mod._compile_tp_gemm1).parameters

    mismatch = []
    for k, want in CTRL_CFG.items():
        if k not in fused_sig or k not in gemm_sig:
            mismatch.append(f"{k}: not a parameter of both compile paths")
            continue
        # What the fused kernel will actually be compiled with. run() passes
        # num_waves from the gather object and everything else from runner.cfg,
        # falling back to the signature default.
        if k == "num_waves":
            eff = int(g.num_waves)
        elif k in runner.cfg:
            eff = runner.cfg[k]
        else:
            eff = fused_sig[k].default
        if eff != want:
            mismatch.append(f"{k}: fused uses {eff!r}, control asserts {want!r}")
    if int(runner.sort_block_m) != int(sort_block_m):
        mismatch.append(
            f"sort_block_m: fused uses {runner.sort_block_m}, control {sort_block_m}"
        )
    if float(runner.swiglu_limit) != float(swiglu_limit):
        mismatch.append(
            f"swiglu_limit: fused uses {runner.swiglu_limit}, control {swiglu_limit}"
        )
    # The push half: the standalone gather kernel and the push embedded in the
    # fused kernel are compiled from the same three knobs, all read off ``g``.
    if int(g.producer_blocks) % int(g.tp_size) != 0:
        mismatch.append("producer_blocks not divisible by tp_size")
    if mismatch:
        raise SystemExit(
            "STOP: Arm A and Arm B cannot be equalised:\n  " + "\n  ".join(mismatch)
        )

    return dict(
        model_dim=int(runner.model_dim),
        inter_dim=int(runner.inter_dim),
        experts=int(runner.experts),
        total_rows=int(g.rows),
        gather=True,
        sort_block_m=int(sort_block_m),
        swiglu_limit=float(swiglu_limit),
        **CTRL_CFG,
    )


def _ctrl_arm_a(g, runner, gemm_kwargs, x_q, x_scale, sids, seids, nvalid, max_sorted):
    """Two launches: standalone P2P push, then standalone GEMM1 over the slab.

    ``g`` is Arm A's OWN TPActivationGather, not the one inside ``runner``.
    ``emit_ticket_and_roles`` recovers a launch generation by dividing a shared
    monotonic counter by the launch's grid size, so two kernels with different
    grids (33 for the standalone push, num_cu*grid_mult=1024 for the fused one)
    cannot share ``entry_count``: the derived gate epoch stops matching the
    host-derived ``launch_epoch`` and every rank hangs on the gate. Verified by
    hanging the 8-rank job for 12 minutes with a single shared gather object.
    """
    s = {}

    def push():
        s["parity"] = g.current_parity()
        g.gather(x_q, x_scale)

    def gemm1():
        parity = s["parity"]
        trb = runner._tile_row_base(
            int(max_sorted) // runner.sort_block_m + 64, x_q.device
        )
        s["out"] = run_tp_gemm1(
            x=g.rx_x[parity],
            scale_x=g.rx_scale[parity],
            w=runner.w,
            scale_w=runner.w_scale,
            tile_row_base=trb,
            expert_ids=seids,
            sorted_token_ids=sids,
            num_valid_ids=nvalid,
            max_sorted=int(max_sorted),
            **gemm_kwargs,
        )

    t = _timed([push, gemm1])
    return t, s["out"]


def _ctrl_arm_b(runner, x_q, x_scale, sids, seids, nvalid, max_sorted):
    """One launch: push and GEMM1 in the same kernel."""
    s = {}

    def fused():
        s["out"] = runner.run(
            x_q=x_q,
            x_scale=x_scale,
            sorted_token_ids=sids,
            expert_ids=seids,
            num_valid_ids=nvalid,
            max_sorted=int(max_sorted),
        )

    t = _timed([fused])
    return t, s["out"]


def _routed_mask(sids, nvalid, max_sorted, m_logical, device):
    """Rows stage1 actually writes. Padding rows are uninitialized memory.

    Two conditions, both necessary. The token-id test drops the padding slots
    moe_sorting inserts to round each expert up to sort_block_m. The row <
    num_valid test drops the tail of the sorted_token_ids buffer past the last
    valid slot: moe_sorting does not fill it with the padding sentinel, so it
    holds whatever the allocator returned, and for m_local >= 128 a large
    fraction of that junk passes the token-id test by accident. Those rows are
    never written by either arm, and comparing them compares two uninitialized
    torch.empty buffers.
    """
    # moe_sorting returns a 2-element tensor: [0] is the valid row count,
    # [1] is the capacity-overflow status. .item() on the whole thing throws.
    n_valid = int(nvalid[0].item())
    keep = torch.zeros(int(max_sorted), dtype=torch.bool, device=device)
    n = min(int(sids.shape[0]), int(max_sorted), n_valid)
    keep[:n] = (sids[:n].to(torch.int64) & 0x00FFFFFF) < int(m_logical)
    return keep, n_valid


def _ctrl_bitcheck(a, b, keep, max_sorted):
    """Bit-identity of payload + mx scale over routed rows. Returns a report."""
    (out_a, os_a), (out_b, os_b) = a, b
    pa = out_a.view(torch.uint8)[: int(max_sorted)][keep]
    pb = out_b.view(torch.uint8)[: int(max_sorted)][keep]
    sa = os_a.view(torch.uint8).reshape(-1)
    sb = os_b.view(torch.uint8).reshape(-1)
    cols = os_b.shape[-1] if os_b.dim() == 2 else None
    if cols is None:
        raise SystemExit("STOP: unexpected out_scale rank from the fused arm")
    sa = sa[: sa.numel() // cols * cols].reshape(-1, cols)[: int(max_sorted)][keep]
    sb = sb[: sb.numel() // cols * cols].reshape(-1, cols)[: int(max_sorted)][keep]
    p_bad = int((pa != pb).sum())
    s_bad = int((sa != sb).sum())
    return {
        "rows": int(keep.sum()),
        "payload_bytes": pa.numel(),
        "payload_mismatch": p_bad,
        "scale_bytes": sa.numel(),
        "scale_mismatch": s_bad,
        "ok": p_bad == 0 and s_bad == 0,
    }


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
    # Two production operators that differ in exactly one compile-time flag:
    # which side triggers the cross-rank movement. Same GEMM1, same tile config,
    # same bytes to the same offsets -- only the direction differs. Both
    # constructors allocate symmetric memory, so the order matters and must be
    # the same on every rank.
    fused = TPMoEStage1(max_tok_per_rank=max(sizes), pull=True, **common)
    fused_push = TPMoEStage1(max_tok_per_rank=max(sizes), pull=False, **common)

    # Arm A of the controlled comparison needs its own symmetric buffers and its
    # own launch-ticket counter (see _ctrl_arm_a). This is a collective
    # allocation: constructed once here, with identical arguments on every rank.
    from aiter.ops.flydsl.kernels.mega_moe.tp_gather import TPActivationGather

    # The controlled two-launch-vs-one-launch comparison is a push-side result
    # and stays that way, so it remains comparable with the numbers already
    # recorded in the phase-2 conclusion. Pull is compared against push
    # separately, through the two full-forward arms.
    runner = fused_push._fused
    gather_a = TPActivationGather(
        model_dim=model_dim,
        tp_size=runner.gather.tp_size,
        tp_rank=runner.gather.tp_rank,
        max_tok_per_rank=max(sizes),
        device=device,
        num_waves=runner.gather.num_waves,
        producer_blocks=runner.gather.producer_blocks,
        double_buffer=runner.gather.slots == 2,
    )
    if (
        gather_a.rows,
        gather_a.num_waves,
        gather_a.producer_blocks,
        gather_a.slots,
    ) != (
        runner.gather.rows,
        runner.gather.num_waves,
        runner.gather.producer_blocks,
        runner.gather.slots,
    ):
        raise SystemExit("STOP: Arm A's gather does not match the fused push half")
    gemm_kwargs = _assert_same_config(
        runner, fused.sort_block_m, NETWORK["swiglu_limit"]
    )

    bf16_row = model_dim * 2
    fp8_row = model_dim + model_dim // 32

    if rank == 0:
        print(
            f"world={world} model_dim={model_dim} inter_dim={inter_dim} "
            f"experts={experts} topk={topk} iters={args.iters} "
            f"max_tok_per_rank={max(sizes)}"
        )
        print(f"gather row bytes: bf16={bf16_row}  fp8+scale={fp8_row}")
        print("controlled-arm config (identical for both arms):")
        print(f"  gemm1: {gemm_kwargs}")
        print(
            f"  push : num_waves={gather_a.num_waves} "
            f"producer_blocks={gather_a.producer_blocks} slots={gather_a.slots} "
            f"total_rows={gather_a.rows}"
        )
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
        fp = _median_stages(
            lambda: _stages_fused(fused_push, x, wts, ids), args.iters, args.warmup
        )
        merged = _max_across_ranks(b + p + f + fp, device)
        b, p, f, fp = merged[:8], merged[8:16], merged[16:24], merged[24:]

        # ---- controlled arm: same GEMM1, two launches vs one ---------------
        x_q_c, x_scale_c = fused.quantize(x)
        meta = torch.empty((m_local, 2 * fused.topk), dtype=torch.int32, device=device)
        meta[:, : fused.topk] = ids
        meta[:, fused.topk :] = wts.view(torch.int32)
        meta_g = fused._all_gather_one(meta)
        ids_g = meta_g[:, : fused.topk].contiguous()
        wts_g = meta_g[:, fused.topk :].contiguous().view(torch.float32)
        sids, _, seids, nvalid = fused._sort(ids_g, wts_g)
        sbm = fused.sort_block_m
        max_sorted = -(-int(sids.shape[0]) // sbm) * sbm
        keep, n_valid = _routed_mask(sids, nvalid, max_sorted, m_global, device)

        # Correctness first: a timing comparison between two arms that do not
        # compute the same thing is meaningless.
        _, out_a = _ctrl_arm_a(
            gather_a,
            runner,
            gemm_kwargs,
            x_q_c,
            x_scale_c,
            sids,
            seids,
            nvalid,
            max_sorted,
        )
        _, out_b = _ctrl_arm_b(
            runner, x_q_c, x_scale_c, sids, seids, nvalid, max_sorted
        )
        torch.cuda.synchronize()
        chk = _ctrl_bitcheck(out_a, out_b, keep, max_sorted)
        chk_all = _max_across_ranks(
            [chk["payload_mismatch"], chk["scale_mismatch"]], device
        )

        ca = _median_stages(
            lambda: _ctrl_arm_a(
                gather_a,
                runner,
                gemm_kwargs,
                x_q_c,
                x_scale_c,
                sids,
                seids,
                nvalid,
                max_sorted,
            )[0],
            args.iters,
            args.warmup,
        )
        cb = _median_stages(
            lambda: _ctrl_arm_b(
                runner, x_q_c, x_scale_c, sids, seids, nvalid, max_sorted
            )[0],
            args.iters,
            args.warmup,
        )
        cmerged = _max_across_ranks(ca + cb, device)
        ca, cb = cmerged[:4], cmerged[4:]

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
        print("  fused path (pull vs push: identical apart from which side")
        print("  triggers the transfer)")
        print(
            f"  {'stage':<16}{'pull dev':>12}{'pull host':>12}"
            f"{'push dev':>12}{'push host':>12}"
        )
        fused_names = ["local quantize", "metadata coll", "moe_sorting", "fused kernel"]
        for i, name in enumerate(fused_names):
            print(
                f"  {name:<16}{f[i]:>12.4f}{f[i + 4]:>12.4f}"
                f"{fp[i]:>12.4f}{fp[i + 4]:>12.4f}"
            )
        tot_fp = sum(fp[:4])
        print(
            f"  {'TOTAL':<16}{tot_f:>12.4f}{sum(f[4:]):>12.4f}"
            f"{tot_fp:>12.4f}{sum(fp[4:]):>12.4f}"
        )
        print(
            f"  kernel only: pull {f[3] * 1e3:.1f}us  push {fp[3] * 1e3:.1f}us  "
            f"delta {(f[3] - fp[3]) * 1e3:+.1f}us"
        )
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
        print("  controlled arm (identical GEMM1, identical bytes; only the")
        print("  launch boundary differs)")
        print(
            f"  bit-check over {chk['rows']} routed rows "
            f"(sorted={int(sids.shape[0])} num_valid={n_valid} "
            f"max_sorted={max_sorted}): "
            f"payload {chk['payload_bytes']}B mismatch={int(chk_all[0])}  "
            f"scale {chk['scale_bytes']}B mismatch={int(chk_all[1])}  "
            f"-> {'IDENTICAL' if max(chk_all) == 0 else 'DIFFER'}"
        )
        tot_a = ca[0] + ca[1]
        print(f"  {'stage':<24}{'dev':>12}{'host':>12}")
        print(f"  {'A push (launch 1)':<24}{ca[0]:>12.4f}{ca[2]:>12.4f}")
        print(f"  {'A gemm1 (launch 2)':<24}{ca[1]:>12.4f}{ca[3]:>12.4f}")
        print(f"  {'A TOTAL':<24}{tot_a:>12.4f}{ca[2] + ca[3]:>12.4f}")
        print(f"  {'B fused (1 launch)':<24}{cb[0]:>12.4f}{cb[1]:>12.4f}")
        print(f"  delta B-A: {cb[0] - tot_a:+.4f} ms   ratio A/B: {tot_a / cb[0]:.3f}x")
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
