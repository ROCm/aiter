# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Accuracy and performance test for the TP MoE layer: AllGather + GEMM1 + GEMM2 + ReduceScatter.

Parallelization model (tensor parallel with sequence-parallel activations)
-------------------------------------------------------------------------
Experts are *replicated* across the TP group; only ``inter_dim`` is sharded::

    w1_r : [E, 2 * inter_dim/TP, model_dim]      (gate rows then up rows)
    w2_r : [E, model_dim, inter_dim/TP]

Activations enter sequence-parallel, so rank ``r`` owns global token rows
``[r*m, (r+1)*m)`` where ``m = M / TP``.  One layer is therefore::

    (1) AG  : x_all[M, H]     = AllGather_token(x_local[m, H])
    (2) G1  : h[M*topk, I_r]  = act(gather(x_all) @ w1_r^T)
    (3) G2  : p[M*topk, H]    = h @ w2_r^T
    (4) RD  : q[M, H]         = sum_k topk_weight[t,k] * p[t,k]        (local)
    (5) RS  : y_local[m, H]   = sum_over_ranks(q)[r*m : (r+1)*m]       (cross-rank)

Steps (2)-(4) are exactly what ``op_tests/test_moe_2stage.py`` measures on a
single GPU with ``-dim <model_dim>,<inter_dim/TP>``: that test *is* the local
compute of one TP rank, sitting between the AllGather and the ReduceScatter.

What this file provides
-----------------------
* ``TorchReferenceTpMoe``  -- torch reference built from ``torch_moe_stage1`` /
  ``torch_moe_stage2``, i.e. the same reference ``test_moe_2stage.py`` uses.
* ``SplitTpMoe``           -- the control group.  Runs the steps in the order
  the fused kernel will, each one timed on its own::

      route AG -> SORT | quant -> AG -> GEMM1 -> GEMM2 -> RS
                        `--------- one fused launch ---------'

  The sort is hoisted out in front because it will not be fused: it needs only
  the routing metadata (topk*8 bytes/token, ~1/45 of the activation payload), so
  it can run while the activations are still on the wire.  ``fused_moe_2stages``
  already takes the sort results as plain arguments, so hoisting is a call-site
  change, not a reimplementation -- the GEMMs stay exactly the tuned kernels
  ``test_moe_2stage.py`` benchmarks.  ``fusable_us`` sums only the legs one
  fused launch replaces, and that is the number to compare against, not e2e.
* ``MegaMoeTP``            -- the fused implementation, a thin adapter over
  ``aiter.ops.flydsl.mega_moe_tp``.  It keeps the same tuned GEMM1/GEMM2 and replaces
  everything around them: the activation is MXFP4-quantized *before* the
  AllGather (3.77x less wire traffic), the AllGather itself is one P2P push
  kernel straight into every peer's arena, GEMM2 accumulates into that arena,
  and the ReduceScatter is a publish + pull-reduce pair over the same mapping.
  ``--impl both`` runs it next to the unfused chain and gates it on both the
  torch reference and the unfused output.

Only a4w4 (MXFP4 activation x MXFP4 weight, ``QuantType.per_1x32``) is wired up
for now.

Making the control group the thing it claims to be
--------------------------------------------------
A speedup is only worth the baseline it is measured against, and this one
started out measuring a baseline no serving stack would ship.  Three things
were wrong with it, all of them inflating the split side:

* **It called RCCL.**  At TP8 decode sizes these collectives are pure latency
  -- kilobytes of payload -- and RCCL's floor on gfx950 is ~18us of *device*
  time per call against ~8us for aiter's one-shot P2P kernels, the ones
  ``tensor_model_parallel_all_gather(use_custom=True)`` exists to reach.  That
  is a ~2.3x handicap per collective before any fusion happens.  ``--comm``
  now selects the backend.  It does not simply default to the one-shot path:
  those kernels also carry 2-3x the *host* dispatch, so they lose in eager and
  win under graph replay, and ``auto`` picks whichever is ahead in the regime
  being measured so the control group is never the slower of the two.
* **It timed staging, not communication.**  The routing AllGather packed
  ids/weights into a staging row and unpacked them again, and the payload
  AllGather did the same for the MXFP4 activation and its scales: four extra
  launches each.  At dsv4 M=8 the routing leg reported ~59us and the payload
  leg ~65us; measured on device against the one-shot backend the same work is
  7.5us and 9.7us.  The packs were never the layer's work (a router emits its
  top-k straight into wire format) and the payload halves never needed packing
  at all.
* **It was eager while the fused side was captured.**  ``--graph`` used to
  capture only the fused path, so every launch the baseline dispatched counted
  as fusion win.  Production decode is captured on both sides; ``--graph`` now
  captures both and reports ``graph_speedup``, which is the only number here
  measured the way the layer actually runs.

The eager columns are kept because they are still what an eager caller sees,
but on this box a single eager launch costs ~14us of host dispatch no matter
what it does, so at small M the eager legs say more about the harness than
about the layer.  Read ``graph_speedup`` for the claim and the eager legs for
where the launches went.

Usage
-----
No wrapper and no exported variables: every environment variable the tuned a4w4
path needs is set at the top of this file, and the repo root is put on
``sys.path`` there too, so the tuned kernels are reached by running the file::

    # all four models, default token sweep
    torchrun --nproc_per_node=8 op_tests/multigpu_tests/test_mega_moe_TP.py

    # one model, accuracy only
    torchrun --nproc_per_node=8 op_tests/multigpu_tests/test_mega_moe_TP.py \
        --models kimi3 --tokens 8 32 128 --no-perf

The single-GPU equivalent of the GEMM block -- the same tuned kernels, driven
through ``fused_moe`` -- is::

    python op_tests/test_moe_2stage.py -q 4 -dim 6144,256 -e 257 -k 9 \
        -a silu -s f -p t -t 8 64 512 4096 --no-flydsl-csv --kernel

with ``-dim <model_dim>,<inter_dim/TP>`` taken from ``MODELS`` below.

The activation wire format is not a free choice: quantizing before the
AllGather is only legal where GEMM1 accepts a quantized activation, and the
``_f16in`` variant that the tuner picks at small M reads raw BF16 and ignores
the A buffers entirely.  ``_TunedPlan.ag_wire`` derives this per bucket, and the
``ag_wire`` column reports it.  On the current CSVs the crossover is M=128 for
kimi3, M=256 for glm5, and M=8 for dsv3.

Known environment limitation
----------------------------
Shapes whose tuned GEMM2 is a CK kernel (e.g. dsv3 at M>=128) need
``module_moe_ck2stages_fp4x2_fp4x2_*``, which does not compile against the CK
headers shipped with ROCm 7.1.0 -- ``blockwise_gemm_pipeline_xdlops_b_preshuffle
_mx_moe_gufusion_v3.hpp`` raises "non-type template argument is not a constant
expression".  Those cases are detected during the rank-0 JIT warmup and skipped
across all ranks rather than aborting the sweep.

Do NOT set ``AITER_CONFIG_FMOE``: left unset, aiter merges every
``aiter/configs/model_configs/*tuned_fmoe*.csv``, which is what makes all four
model shapes hit tuned kernels in one process.  Setting it pins one CSV and the
other three models fall back to untuned heuristics.
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import sys
from dataclasses import dataclass, replace
from typing import ClassVar

# ---------------------------------------------------------------------------
# Environment and search path -- must run before ``import aiter``
# ---------------------------------------------------------------------------
# Running this file is meant to be enough: no wrapper script, no exported
# variables.  Everything the tuned a4w4 path needs is set here with
# ``setdefault``, so an explicit value from the caller still wins.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.isdir(os.path.join(_REPO_ROOT, "aiter")) and _REPO_ROOT not in sys.path:
    # Prefer this checkout over any installed amd-aiter wheel, which on this box
    # points at a different tree.  Replaces ``PYTHONPATH=$PWD``.
    sys.path.insert(0, _REPO_ROOT)

for _key, _value in {
    # Read at *import* time by aiter/ops/triton/gluon/__init__.py, which raises
    # when the installed triton is older than 3.6 (this box has 3.4).
    "AITER_USE_SYSTEM_TRITON": "1",
    # Resolve SiTUv2 to a4w4 (fp4x2) instead of falling back to a16w4
    # (fused_moe.py:1077).
    "AITER_SITUV2_A4W4": "1",
    # GEMM2 emits fp8 per-route partials plus a separate reduction, which is the
    # configuration the tuned CSV rows were measured under (fused_moe.py:2468).
    "AITER_FLYDSL_STAGE2_FP8": "1",
    # The 256 default would select bf16 activations below M=256
    # (fused_moe.py:1060), i.e. a16w4 rather than the a4w4 under test.
    "AITER_BF16_FP8_MOE_BOUND": "0",
}.items():
    os.environ.setdefault(_key, _value)

# Deliberately NOT set here:
#   AITER_CONFIG_FMOE        Pinning one CSV sends the other three models to
#                            untuned heuristics.  Left unset, aiter globs and
#                            merges every ``model_configs/*tuned_fmoe*.csv``, so
#                            all four shapes hit their own tuned rows in one
#                            process.  Each shape names its file in
#                            ``ModelShape.tuned_csv`` and ``check_tuned_csvs()``
#                            asserts they are all present at startup.
#   AITER_MOE_EXPERT_BALANCE Read only by op_tests/test_moe_2stage.py, never by
#                            aiter.  The equivalent here is ``--route``, which
#                            already defaults to ``balanced``.

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import aiter  # noqa: E402
from aiter import dtypes  # noqa: E402
from aiter.dist.communication_op import (  # noqa: E402
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from aiter.fused_moe import (  # noqa: E402
    fused_moe_2stages,
    fused_topk,
    get_2stage_cfgs,
    get_padded_M,
    moe_sorting,
    stage2_uses_route_reduce,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.jit.core import AITER_CONFIGS  # noqa: E402
from aiter.jit.utils.chip_info import get_cu_num, get_gfx  # noqa: E402
from aiter.ops.flydsl.moe_common import (  # noqa: E402
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
    GateMode,
)
from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP as MegaMoeTPEngine  # noqa: E402
from aiter.ops.flydsl.mega_moe_tp import (  # noqa: E402
    MegaMoeTPConfig,
    _gemm1_takes_prequantized_fp4,
    mega_moe_tp_supported,
)
from aiter.ops.flydsl.mxfp4_kname import parse_g2_kname_any  # noqa: E402
from aiter.ops.quant import get_hip_quant  # noqa: E402
from aiter.ops.shuffle import shuffle_weight  # noqa: E402
from aiter.utility import fp4_utils  # noqa: E402

logger = logging.getLogger("aiter")

SUPPORTED_GFX = ("gfx950",)

# a4w4: MXFP4 activation x MXFP4 weight, 1x32 E8M0 microscales.
QUANT_TYPE = aiter.QuantType.per_1x32
AQ_DTYPE = dtypes.fp4x2
WQ_DTYPE = dtypes.fp4x2


# ---------------------------------------------------------------------------
# Model shapes
# ---------------------------------------------------------------------------
# ``inter_dim`` below is the FULL (unsharded) model intermediate size; the test
# shards it by TP, so the per-rank GEMM sees ``inter_dim // tp``.  Each entry's
# TP8 shard is a shape that already has tuned rows in the referenced CSV, which
# is why the sweep lands on tuned kernels rather than heuristics.
@dataclass(frozen=True)
class ModelShape:
    name: str
    model_dim: int
    inter_dim: int  # full, unsharded
    experts: int
    topk: int
    act_type: aiter.ActivationType
    tuned_csv: str
    # True when ``tuned_csv`` was tuned for a different activation dtype, so the
    # a4w4 sweep will fall back to untuned kernel selection.
    a4w4_untuned: bool = False

    def local_inter_dim(self, tp: int) -> int:
        if self.inter_dim % tp:
            raise ValueError(
                f"{self.name}: inter_dim={self.inter_dim} is not divisible by TP={tp}"
            )
        return self.inter_dim // tp

    def tag(self, tp: int) -> str:
        return (
            f"{self.name} h{self.model_dim} i{self.inter_dim}/{tp}="
            f"{self.local_inter_dim(tp)} e{self.experts} k{self.topk}"
        )


MODELS: dict[str, ModelShape] = {
    # kimik3_a4w4_tuned_fmoe.csv: 3584 x 384 x 896 x 16, ActivationType.Situv2
    "kimi3": ModelShape(
        name="kimi3",
        model_dim=3584,
        inter_dim=3072,
        experts=896,
        topk=16,
        act_type=aiter.ActivationType.Situv2,
        tuned_csv="kimik3_a4w4_tuned_fmoe.csv",
    ),
    # dsv3_fp4_tuned_fmoe.csv: 7168 x {256,512,2048} x {256,257} x {8,9}
    "dsv3": ModelShape(
        name="dsv3",
        model_dim=7168,
        inter_dim=2048,
        experts=256,
        topk=8,
        act_type=aiter.ActivationType.Silu,
        tuned_csv="dsv3_fp4_tuned_fmoe.csv",
    ),
    # dsv4_fp8fp4_tuned_fmoe.csv: 7168 x {384,512,768,1536,3072} x 384 x 6.
    # That CSV is a8w4 (fp8 activation); the a4w4 sweep here has no tuned rows.
    "dsv4": ModelShape(
        name="dsv4",
        model_dim=7168,
        inter_dim=3072,
        experts=384,
        topk=6,
        act_type=aiter.ActivationType.Silu,
        tuned_csv="dsv4_fp8fp4_tuned_fmoe.csv",
        a4w4_untuned=True,
    ),
    # glm5_fp4_tuned_fmoe.csv: 6144 x {256,512,1024,2048} x 257 x 9
    "glm5": ModelShape(
        name="glm5",
        model_dim=6144,
        inter_dim=2048,
        experts=257,
        topk=9,
        act_type=aiter.ActivationType.Silu,
        tuned_csv="glm5_fp4_tuned_fmoe.csv",
    ),
}


# ---------------------------------------------------------------------------
# Distributed plumbing
# ---------------------------------------------------------------------------
@dataclass
class DistCtx:
    rank: int
    world: int
    device: torch.device
    #: Whether aiter's parallel state came up, i.e. whether the one-shot P2P
    #: collectives are reachable at all.  Empty string when they are; otherwise
    #: the reason, which the backend selection prints once.
    custom_comm_error: str = ""

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_dist(want_custom_comm: bool = True) -> DistCtx:
    """Bring up the process group, preferring aiter's parallel state.

    ``init_dist_env`` is the entry point a serving stack uses: it initializes
    the same NCCL process group *and* the CustomAllreduce IPC arena that backs
    the one-shot P2P AllGather / ReduceScatter.  Without it those kernels are
    simply unreachable and the only collective available is RCCL -- which at
    decode sizes is ~2.3x slower per call on device (see ``TpCollectives``).

    Falls back to a bare NCCL group when the arena cannot be created, so a box
    without working IPC still runs the sweep instead of aborting it.
    """
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    err = ""
    if want_custom_comm and not dist.is_initialized():
        try:
            from aiter.ops.communication import init_dist_env

            if rank == 0:
                # Announced because this step can *hang* rather than fail on
                # boxes where the IPC rendezvous inside ``init_custom_ar`` does
                # not complete (seen on torch 2.8 + ROCm 7.1 here, fine on
                # torch 2.13). A hang with no output is indistinguishable from
                # a slow JIT build; a hang after this line is diagnosable, and
                # ``--comm rccl`` skips the step entirely.
                print(
                    "[TP-MOE] bringing up aiter parallel state for the "
                    "one-shot collectives (--comm rccl skips this)",
                    flush=True,
                )
            init_dist_env(world, rank, local_rank=local_rank)
        except Exception as exc:  # noqa: BLE001 - degrade to RCCL, never abort
            err = f"{type(exc).__name__}: {exc}"
    elif not want_custom_comm:
        err = "disabled by --comm rccl"
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
        if want_custom_comm and not err:
            err = "aiter parallel state did not initialize"
    torch.set_default_device(device)
    return DistCtx(rank=rank, world=world, device=device, custom_comm_error=err)


def cleanup_dist() -> None:
    try:
        from aiter.ops.communication import destroy_dist_env

        destroy_dist_env()
    except Exception as exc:  # noqa: BLE001 - teardown must not mask a failure
        logger.debug("[TP-MOE] destroy_dist_env: %s", exc)
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    torch.cuda.synchronize()
    dist.barrier()


def _free_gib(device) -> float:
    free, _total = torch.cuda.mem_get_info(device)
    return free / (1 << 30)


def _reduce_scalar(value: float, device, op) -> float:
    t = torch.tensor(float(value), dtype=torch.float32, device=device)
    dist.all_reduce(t, op=op)
    return float(t.item())


def rank_max(value: float, device) -> float:
    return _reduce_scalar(value, device, dist.ReduceOp.MAX)


def rank_mean(value: float, device) -> float:
    return _reduce_scalar(value, device, dist.ReduceOp.SUM) / dist.get_world_size()


def all_ranks_ok(ok: bool, device) -> bool:
    """Agree across ranks on whether to proceed.

    A rank that fails (typically OOM) must not simply skip ahead: its peers are
    already inside the next collective and would hang. Every conditional step
    therefore votes first.
    """
    t = torch.tensor(int(ok), dtype=torch.int32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


def time_us(
    fn, *, iters: int, warmup: int, device, rounds: int = 3
) -> tuple[float, float]:
    """Return (mean_us, max_us) across ranks for one callable.

    Collectives are involved, so every rank must enter the timed region
    together -- hence the barrier before each round.

    The measurement is repeated ``rounds`` times and the fastest round wins.
    Interference from other tenants on the node only ever *adds* time, and a
    single spike is enough to make a whole loop meaningless (we have seen a 4x
    outlier on an otherwise 320us case), so the minimum is the right estimator
    here.  Rounds are reduced across ranks before being compared, so the
    returned mean and max always come from the same round.
    """
    for _ in range(warmup):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    best_mean = best_max = float("inf")
    for _ in range(max(1, rounds)):
        barrier()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        local = start.elapsed_time(end) / iters * 1000.0
        round_mean = rank_mean(local, device)
        round_max = rank_max(local, device)
        if round_max < best_max:
            best_mean, best_max = round_mean, round_max
    return best_mean, best_max


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
@dataclass
class TpMoeWeights:
    """Per-rank inter-dim shard of one MoE layer, in both kernel and reference form.

    The "full" weight is defined as the concatenation of the TP shards, so
    summing every rank's partial output reproduces the unsharded layer.  The
    full tensor is never materialized (it would be tens of GiB).
    """

    shape: ModelShape
    tp_size: int
    local_inter_dim: int
    # Kernel-facing: preshuffled MXFP4 + shuffled E8M0 scales.
    w1: torch.Tensor  # [E, 2*I_r, H/2] fp4x2, shuffle_weight (16,16)
    w1_scale: torch.Tensor
    w2: torch.Tensor  # [E, H, I_r/2] fp4x2, shuffle_weight (16,16)
    w2_scale: torch.Tensor
    # Reference-facing: unshuffled MXFP4 + raw E8M0 scales.
    w1_ref: torch.Tensor
    w1_scale_ref: torch.Tensor
    w2_ref: torch.Tensor
    w2_scale_ref: torch.Tensor


def _quantize_experts_chunked(
    experts: int,
    rows: int,
    cols: int,
    magnitude: float,
    seed: int,
    device,
    chunk_bytes: int = 512 << 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build and MXFP4-quantize an ``[experts, rows, cols]`` weight, chunk by chunk.

    Materializing the whole bf16 tensor first would need ``experts*rows*cols*2``
    bytes plus the quantizer's fp32 temporaries, which for dsv4's w1 shard is
    ~12 GiB.  Quantizing a slice of experts at a time caps the peak instead, and
    is bit-identical to the one-shot path because each chunk is reseeded.

    Returns ``(qt[experts, rows, cols//2] fp4x2, scale[experts*rows, cols//32])``
    -- note the quantizer flattens the leading two dims of the scale.
    """
    per_expert = rows * cols * 2
    chunk = max(1, min(experts, chunk_bytes // max(per_expert, 1)))
    quant = aiter.get_torch_quant(QUANT_TYPE)
    qt = torch.empty((experts, rows, cols // 2), dtype=torch.uint8, device=device)
    scale: torch.Tensor | None = None
    gen = torch.Generator(device=device)
    for start in range(0, experts, chunk):
        n = min(chunk, experts - start)
        # Reseed per chunk so the result does not depend on the chunk size.
        gen.manual_seed(seed + start)
        w = torch.randn(
            (n, rows, cols), dtype=dtypes.bf16, device=device, generator=gen
        )
        w.mul_(magnitude)
        q, s = quant(w, quant_dtype=WQ_DTYPE)
        del w
        qt[start : start + n] = q.view(n, rows, cols // 2).view(torch.uint8)
        if scale is None:
            scale = torch.empty(
                (experts * rows, s.shape[-1]), dtype=s.dtype, device=device
            )
        scale[start * rows : (start + n) * rows] = s.view(n * rows, -1)
        del q, s
        torch.cuda.empty_cache()
    assert scale is not None
    return qt.view(WQ_DTYPE), scale


def build_sharded_weights(
    shape: ModelShape, ctx: DistCtx, tp_size: int, seed: int
) -> TpMoeWeights:
    """Generate this rank's inter-dim shard directly, quantize, and preshuffle."""
    inter_r = shape.local_inter_dim(tp_size)
    model_dim, experts = shape.model_dim, shape.experts
    base = seed + 1_000_000 * ctx.rank

    # w1 shard rows are [gate_r ; up_r], i.e. exactly the use_g1u1 layout with
    # inter_dim = inter_r.  From the kernel's point of view this is simply a MoE
    # whose intermediate size is inter_r; no special casing is needed anywhere.
    w1_qt, w1_scale = _quantize_experts_chunked(
        experts, 2 * inter_r, model_dim, model_dim**-0.25, base, ctx.device
    )
    w2_qt, w2_scale = _quantize_experts_chunked(
        experts, model_dim, inter_r, inter_r**-0.25, base + 7, ctx.device
    )
    torch.cuda.empty_cache()

    # a4w4 + preshuffle: test_moe_2stage.py:356-360 takes the generic
    # shuffle_weight((16,16)) + e8m0_shuffle path (NOT shuffle_weight_a16w4,
    # which only applies when the activation is bf16/fp16/fp8).
    return TpMoeWeights(
        shape=shape,
        tp_size=tp_size,
        local_inter_dim=inter_r,
        w1=shuffle_weight(w1_qt, layout=(16, 16)),
        w1_scale=fp4_utils.e8m0_shuffle(w1_scale),
        w2=shuffle_weight(w2_qt, layout=(16, 16)),
        w2_scale=fp4_utils.e8m0_shuffle(w2_scale),
        w1_ref=w1_qt,
        w1_scale_ref=w1_scale,
        w2_ref=w2_qt,
        w2_scale_ref=w2_scale,
    )


# ---------------------------------------------------------------------------
# Inputs / routing
# ---------------------------------------------------------------------------
@dataclass
class TpMoeInputs:
    """One sequence-parallel activation shard plus its (already gathered) routing."""

    x_local: torch.Tensor  # [m, H] bf16, this rank's token shard
    topk_weights_local: torch.Tensor  # [m, topk] f32
    topk_ids_local: torch.Tensor  # [m, topk] i32
    #: ids and (bit-cast) weights side by side in one [m, 2*topk] int32 row, so
    #: the routing metadata rides a single collective instead of two.
    #:
    #: This buffer is built here, with the routing, and not inside the layer.
    #: Packing it per call used to cost two extra launches on every timed
    #: routing AllGather, charged to the collective even though it is the
    #: router's output format and not the layer's work. A
    #: real router writes its top-k straight into whatever the layer is about
    #: to send; the layer's job starts at the collective.
    route_meta_local: torch.Tensor  # [m, 2*topk] i32
    global_tokens: int
    local_tokens: int


def make_inputs(
    shape: ModelShape,
    ctx: DistCtx,
    tp_size: int,
    global_tokens: int,
    seed: int,
    route: str,
) -> TpMoeInputs:
    if global_tokens % tp_size:
        raise ValueError(
            f"global tokens={global_tokens} must be divisible by TP={tp_size}"
        )
    m = global_tokens // tp_size
    gen = torch.Generator(device=ctx.device).manual_seed(seed + ctx.rank)
    x = torch.randn(
        (m, shape.model_dim), dtype=dtypes.bf16, device=ctx.device, generator=gen
    )
    if route == "balanced":
        # Mirrors AITER_MOE_EXPERT_BALANCE in test_moe_2stage.py:157-164: walk
        # the expert ids round-robin so every expert gets a near-equal share.
        # At 896 experts / topk 16 the routing distribution dominates runtime,
        # so this is the default for stable perf numbers.
        score = torch.zeros((m, shape.experts), dtype=dtypes.bf16, device=ctx.device)
        start = (ctx.rank * m * shape.topk) % shape.experts
        for token_id in range(m):
            end = start + shape.topk
            if end <= shape.experts:
                score[token_id, start:end] = 1.0
            else:
                score[token_id, start:] = 1.0
                score[token_id, : end - shape.experts] = 1.0
            start = end % shape.experts
    else:
        score = torch.randn(
            (m, shape.experts), dtype=dtypes.bf16, device=ctx.device, generator=gen
        )
    topk_weights, topk_ids = fused_topk(x, score, shape.topk, True)
    topk_weights = topk_weights.contiguous()
    topk_ids = topk_ids.contiguous()
    # The router's wire format, built once with the routing itself -- see
    # TpMoeInputs.route_meta_local for why this does not belong in the layer.
    meta = torch.empty(
        (m, 2 * shape.topk), dtype=torch.int32, device=ctx.device
    ).contiguous()
    meta[:, : shape.topk].copy_(topk_ids)
    meta[:, shape.topk :].copy_(topk_weights.view(torch.int32))
    return TpMoeInputs(
        x_local=x.contiguous(),
        topk_weights_local=topk_weights,
        topk_ids_local=topk_ids,
        route_meta_local=meta,
        global_tokens=global_tokens,
        local_tokens=m,
    )


# ---------------------------------------------------------------------------
# Collective backend
# ---------------------------------------------------------------------------
class TpCollectives:
    """The AllGather / ReduceScatter the split baseline runs on.

    Which backend this is is not an implementation detail, it decides whether
    the baseline is the thing the fused kernel actually has to beat.  At TP8
    decode sizes these calls are pure latency -- the payload is kilobytes --
    and on gfx950 the two available backends are nowhere near each other.
    Measured on this box at M=8, dsv4 (H=7168), device time per call (``reps``
    copies captured inside one CUDA graph, so no host dispatch is included)::

        AllGather   bf16          RCCL 18.4us    aiter one-shot  7.8us
        ReduceScatter bf16        RCCL 18.3us    aiter one-shot  8.2us
        AllGather   mxfp4 wire    RCCL 25.2us    aiter one-shot  7.3us

    A serving stack runs the one-shot kernels -- that is exactly what
    ``tensor_model_parallel_all_gather(use_custom=True)`` and
    ``tensor_model_parallel_reduce_scatter(use_custom=True)`` exist for -- so a
    baseline wired straight to ``torch.distributed`` concedes ~2.3x on every
    collective before fusion enters the picture, and any speedup quoted against
    it inherits that.

    The one-shot path is not free, though: its Python wrapper, torch.library
    dispatch and per-call output allocation cost ~26-35us of *host* time per
    call against RCCL's ~11us, so it is behind in eager and ahead under graph
    replay.  ``--comm auto`` resolves that before this object is built (see
    ``main``); by the time a backend arrives here it is already the one that
    wins in the regime being measured.  Within a backend, individual calls
    still fall back to RCCL when the shape is one the kernel declines, which is
    what a serving stack does too -- ``describe()`` reports when that happened
    rather than letting a mixed baseline pass as a pure one.

    Eligibility is the kernel's, not ours: ``should_custom_ag`` wants a
    contiguous input whose byte size is a multiple of 16, and the kernel itself
    only accepts fp32/fp16/bf16.  Byte-identical dtypes are therefore bit-cast
    on the way in and back on the way out -- an MXFP4 payload is just bytes on
    the wire, and a collective that only copies does not care how they are
    interpreted.
    """

    #: Byte-identical float view for dtypes the one-shot kernel rejects.
    #: int32/int16/int64 are already handled inside ``custom_all_gather``;
    #: these are the ones that reach it unconverted.
    _BYTE_VIEW: ClassVar[dict] = {
        torch.uint8: torch.bfloat16,
        torch.int8: torch.bfloat16,
    }

    def __init__(self, backend: str, ctx: DistCtx, tp_size: int):
        self.tp_size = tp_size
        self.device = ctx.device
        self.enabled = backend in ("auto", "custom") and not ctx.custom_comm_error
        self.error = "" if self.enabled else (ctx.custom_comm_error or "")
        #: Set once the first fallback happens, so the report can say the
        #: baseline was not purely one-shot rather than silently mixing.
        self.fell_back = False

    # -- eligibility ------------------------------------------------------
    def _as_float(self, x: torch.Tensor):
        """Return a byte-identical float view, or None if there is not one."""
        want = self._BYTE_VIEW.get(x.dtype)
        if want is None:
            return x
        step = torch.finfo(want).bits // 8 // x.element_size()
        if x.dim() != 2 or x.shape[-1] % step:
            return None
        return x.view(want)

    def _ag_custom_ok(self, x: torch.Tensor) -> bool:
        if not self.enabled or not x.is_contiguous():
            return False
        # should_custom_ag: 16-byte multiple, and the gathered result has to fit
        # the registered IPC arena (it is 1 GiB, so only the largest sweep
        # points come near it).
        return (x.numel() * x.element_size()) % 16 == 0

    def _rs_custom_ok(self, x: torch.Tensor) -> bool:
        if not self.enabled or not x.is_contiguous():
            return False
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return False
        if x.shape[0] % self.tp_size:
            return False
        pack = 16 // x.element_size()
        return x.numel() % (self.tp_size * pack) == 0

    # -- collectives ------------------------------------------------------
    def all_gather(self, x: torch.Tensor, out: torch.Tensor | None = None):
        """Gather ``x[m, ...]`` along dim 0 into ``[m * TP, ...]``.

        ``out`` is only a hint: the one-shot path returns its own buffer (the
        kernel writes into the IPC arena), while the RCCL path writes into the
        preallocated buffer so it is not charged an allocation the one-shot
        path does not pay either.
        """
        if self._ag_custom_ok(x):
            view = self._as_float(x)
            if view is not None:
                try:
                    got = tensor_model_parallel_all_gather(
                        view, use_custom=True, dim=0
                    )
                    return got.view(x.dtype) if view.dtype != x.dtype else got
                except Exception:  # noqa: BLE001 - shape the kernel declines
                    self.fell_back = True
        if out is None:
            out = torch.empty(
                (x.shape[0] * self.tp_size,) + tuple(x.shape[1:]),
                dtype=x.dtype,
                device=x.device,
            )
        dist.all_gather_into_tensor(out, x)
        return out

    def reduce_scatter(self, x: torch.Tensor, out: torch.Tensor | None = None):
        """Sum ``x[M, ...]`` across ranks and keep this rank's ``[M/TP, ...]``."""
        if self._rs_custom_ok(x):
            try:
                return tensor_model_parallel_reduce_scatter(
                    x, use_custom=True, dim=0
                )
            except Exception:  # noqa: BLE001 - shape the kernel declines
                self.fell_back = True
        if out is None:
            out = torch.empty(
                (x.shape[0] // self.tp_size,) + tuple(x.shape[1:]),
                dtype=x.dtype,
                device=x.device,
            )
        dist.reduce_scatter_tensor(out, x)
        return out

    def describe(self) -> str:
        if not self.enabled:
            return f"rccl ({self.error})" if self.error else "rccl"
        return "custom+rccl-fallback" if self.fell_back else "custom"


# ---------------------------------------------------------------------------
# Module 1 / 4: AllGather of the token shard
# ---------------------------------------------------------------------------
class AllGatherTokens:
    """AllGather the sequence-parallel activation shard into the full token set.

    Buffers are preallocated so the timed callable does no allocation, and so
    the same instance can be captured by a CUDA graph later.
    """

    def __init__(
        self,
        shape: ModelShape,
        tp_size: int,
        max_local_tokens: int,
        device,
        comm: TpCollectives | None = None,
    ):
        self.tp_size = tp_size
        self.model_dim = shape.model_dim
        self.topk = shape.topk
        self.comm = comm
        total = max_local_tokens * tp_size
        self._x = torch.empty(
            (total, shape.model_dim), dtype=dtypes.bf16, device=device
        )
        self._w = torch.empty((total, shape.topk), dtype=torch.float32, device=device)
        self._i = torch.empty((total, shape.topk), dtype=torch.int32, device=device)
        # Routing metadata is tiny (topk*8 bytes/token) but must be gathered too:
        # every rank evaluates every global token against its own expert shard.
        # It arrives already packed as ``inputs.route_meta_local``, so this pays
        # one collective for it rather than a second latency floor.
        self._meta = torch.empty(
            (total, 2 * shape.topk), dtype=torch.int32, device=device
        )

    def wire_bytes(self, local_tokens: int) -> int:
        """Bytes this rank receives over the fabric (excluding its own shard)."""
        row = self.model_dim * 2 + self.topk * 8
        return local_tokens * row * (self.tp_size - 1)

    def __call__(self, inputs: TpMoeInputs):
        g, k = inputs.global_tokens, self.topk
        if self.comm is None:
            dist.all_gather_into_tensor(self._x[:g], inputs.x_local)
            dist.all_gather_into_tensor(self._meta[:g], inputs.route_meta_local)
            x, meta = self._x[:g], self._meta[:g]
        else:
            x = self.comm.all_gather(inputs.x_local, out=self._x[:g])
            meta = self.comm.all_gather(
                inputs.route_meta_local, out=self._meta[:g]
            )
        w, i = self._w[:g], self._i[:g]
        i.copy_(meta[:, :k])
        w.view(torch.int32).copy_(meta[:, k:])
        return x, w, i


class QuantWireAllGatherProbe:
    """Sizes the AllGather *collective* at MXFP4 wire width instead of bf16.

    This is deliberately not part of the functional path: ``fused_moe``
    quantizes internally, so handing it prequantized input would double-quantize.
    The probe exists to price the "quantize locally, then AllGather"
    optimization the fused kernel should use -- the wire row shrinks from
    ``H*2`` to ``H/2 + H/32`` bytes (3.77x less traffic), and each rank
    quantizes only its own m rows rather than all M.

    Only the collective is timed.  The quantization is excluded on purpose: in
    the fused design it is absorbed into K1 alongside the AllGather push, so
    charging it as a standalone launch here would measure the wrong thing.  Use
    ``quant_local_us`` for the separate cost of the local quantize.
    """

    def __init__(self, shape: ModelShape, tp_size: int, max_local_tokens: int, device):
        self.tp_size = tp_size
        self.model_dim = shape.model_dim
        # fp4 payload and its E8M0 scales, packed into one contiguous row so the
        # probe pays a single collective's latency -- same accounting as the
        # bf16 AllGatherTokens above.
        self.row_bytes = shape.model_dim // 2 + shape.model_dim // 32
        self._local = torch.empty(
            (max_local_tokens, self.row_bytes), dtype=torch.uint8, device=device
        )
        self._all = torch.empty(
            (max_local_tokens * tp_size, self.row_bytes),
            dtype=torch.uint8,
            device=device,
        )

    def wire_bytes(self, local_tokens: int) -> int:
        return local_tokens * self.row_bytes * (self.tp_size - 1)

    def __call__(self, inputs: TpMoeInputs):
        g, m = inputs.global_tokens, inputs.local_tokens
        dist.all_gather_into_tensor(self._all[:g], self._local[:m])
        return self._all[:g]

    @staticmethod
    def quantize_local(inputs: TpMoeInputs):
        """The local MXFP4 quantize, timed separately from the collective.

        Uses the HIP kernel, not ``get_torch_quant(per_1x32)`` -- the latter is
        the torch *reference* implementation and is ~350us even for one row,
        which would make this number meaningless.
        """
        return get_hip_quant(QUANT_TYPE)(inputs.x_local, quant_dtype=AQ_DTYPE)


# ---------------------------------------------------------------------------
# Modules 2 / 3: GEMM1 and GEMM2 (the kernels test_moe_2stage.py drives)
# ---------------------------------------------------------------------------
def _partial_keyword(fn, *keys: str) -> str:
    """Dig the first matching keyword out of a (possibly nested) functools.partial.

    The MXFP4 port stores the name under ``kernelName1``/``kernelName2``
    (fused_moe.py:2621-2627) while the FlyDSL stage wrappers use plain
    ``kernelName`` (fused_moe.py:3156-3162), so try every spelling.
    """
    seen = 0
    while fn is not None and seen < 8:
        kwargs = getattr(fn, "keywords", None) or {}
        for key in keys:
            if kwargs.get(key):
                return str(kwargs[key])
        fn = getattr(fn, "func", None)
        seen += 1
    return ""


# ---------------------------------------------------------------------------
# Module 4 / 4: ReduceScatter of the partial output
# ---------------------------------------------------------------------------
class ReduceScatterOutput:
    """Sum the per-rank partial outputs across TP and scatter back to the token shard."""

    def __init__(
        self,
        shape: ModelShape,
        tp_size: int,
        max_local_tokens: int,
        device,
        comm: TpCollectives | None = None,
    ):
        self.tp_size = tp_size
        self.model_dim = shape.model_dim
        self.comm = comm
        self._y = torch.empty(
            (max_local_tokens, shape.model_dim), dtype=dtypes.bf16, device=device
        )

    def wire_bytes(self, local_tokens: int) -> int:
        return local_tokens * self.model_dim * 2 * (self.tp_size - 1)

    def __call__(self, partial: torch.Tensor, local_tokens: int) -> torch.Tensor:
        # A GEMM2 output is already contiguous, so this is a host-side no-op
        # rather than a copy; it is kept because both backends require it.
        partial = partial.contiguous()
        out = self._y[:local_tokens]
        if self.comm is None:
            dist.reduce_scatter_tensor(out, partial)
            return out
        return self.comm.reduce_scatter(partial, out=out)


# ---------------------------------------------------------------------------
# Split baseline, ordered exactly like the fused kernel will be
# ---------------------------------------------------------------------------
# The fused kernel collapses quant + AG + GEMM1 + GEMM2 + RS into one launch and
# leaves the sort outside, so the control group runs those same steps in that
# same order with each one timed on its own.  Whatever the fused number gains
# over the sum of these legs is fusion gain; whatever it loses is fusion
# overhead.
#
# The sort is hoisted because it will not be fused.  It needs only the routing
# metadata -- topk*8 bytes per token, ~1/45 of the activation payload at
# kimi3 -- so it can be gathered and sorted while the activations are still in
# flight.  Everything downstream consumes ``sorted_ids`` and friends as plain
# inputs, exactly as ``fused_moe_2stages`` already does.
@dataclass(frozen=True)
class _TunedPlan:
    """Everything the tuned CSV decides for one global token count.

    Resolved once per M and reused, so the timed legs never pay a lookup.
    """

    tokens: int
    metadata: object
    block_m: int
    #: GEMM2 uses the atomic epilogue, so ``moe_sorting`` must zero the output
    #: buffer it hands back (fused_moe.py:1275-1288).
    accumulate: bool
    #: GEMM1 reads a pre-quantized FP4 activation.  False for the ``_f16in``
    #: (inline-quant) variant, which reads raw BF16 and ignores the A buffers
    #: entirely -- handing that one a quantized A faults.
    prequant: bool
    gemm1_kernel: str
    gemm2_kernel: str

    @property
    def ag_wire(self) -> str:
        """Which wire format the AllGather can use at this M.

        Tied to the GEMM1 variant, not chosen freely: quantizing before the
        AllGather is only legal when GEMM1 accepts a quantized activation.
        """
        return "fp4_1x32" if self.prequant else "bf16"


class TunedPlans:
    """Per-M tuned-config lookup, shared by every step of the split baseline."""

    def __init__(self, weights: TpMoeWeights):
        self.weights = weights
        self.shape = weights.shape
        self._cache: dict[int, _TunedPlan] = {}

    def __call__(self, global_tokens: int) -> _TunedPlan:
        hit = self._cache.get(global_tokens)
        if hit is not None:
            return hit
        shape = self.shape
        metadata = get_2stage_cfgs(
            get_padded_M(global_tokens),
            shape.model_dim,
            self.weights.local_inter_dim,
            shape.experts,
            shape.topk,
            dtypes.bf16,
            AQ_DTYPE,
            WQ_DTYPE,
            QUANT_TYPE,
            True,  # use_g1u1
            shape.act_type,
            False,  # doweight_stage1
            0,
            0,
            True,  # is_shuffled
            GateMode.SEPARATED.value,
        )
        kname1 = _partial_keyword(metadata.stage1, "kernelName1", "kernelName")
        kname2 = _partial_keyword(metadata.stage2, "kernelName2", "kernelName")
        if metadata.output_aux:
            # Only the Opus branch needs the name parsed, and only FlyDSL names
            # parse.  Some shapes (dsv3 at M>=128) resolve to a CK stage2, whose
            # name is a different grammar entirely -- those never reach here
            # because CK implies output_aux is False, but guard anyway so a
            # parser change downgrades to a skip instead of a crash.
            accumulate = bool(parse_g2_kname_any(kname2)["atomic"])
        else:
            accumulate = not stage2_uses_route_reduce(metadata.stage2)
        plan = _TunedPlan(
            tokens=global_tokens,
            metadata=metadata,
            block_m=int(metadata.block_m),
            accumulate=accumulate,
            prequant=_gemm1_takes_prequantized_fp4(kname1),
            gemm1_kernel=kname1,
            gemm2_kernel=kname2,
        )
        self._cache[global_tokens] = plan
        return plan


class RouteAllGather:
    """Step 0: gather the routing metadata.  The sort's only input.

    One collective plus the two unpack copies ``moe_sorting`` forces, and
    nothing else.  What used to sit here was a four-copy sandwich -- pack
    ids/weights into a staging row, gather, unpack both halves back out -- and
    it dominated its own measurement: at dsv4 M=8 the leg reported ~59us for
    what is 7.5us of device work on the one-shot backend, the rest being two
    packs, two unpacks and the eager dispatch of five launches instead of
    three.  The packs are gone
    because the router already emits ``route_meta_local`` in wire format (see
    ``TpMoeInputs``); the unpacks stay because ``moe_sorting`` needs two
    separately contiguous tensors, and a strided view of one gathered buffer is
    not that.
    """

    def __init__(
        self,
        shape: ModelShape,
        tp_size: int,
        max_local_tokens: int,
        device,
        comm: TpCollectives | None = None,
    ):
        total = max_local_tokens * tp_size
        self.tp_size = tp_size
        self.topk = shape.topk
        self.row_bytes = shape.topk * 8
        self.comm = comm
        self._meta = torch.empty(
            (total, 2 * shape.topk), dtype=torch.int32, device=device
        )
        self._w = torch.empty((total, shape.topk), dtype=torch.float32, device=device)
        self._i = torch.empty((total, shape.topk), dtype=torch.int32, device=device)

    def wire_bytes(self, local_tokens: int) -> int:
        return local_tokens * self.row_bytes * (self.tp_size - 1)

    def __call__(self, inputs: TpMoeInputs):
        g, k = inputs.global_tokens, self.topk
        if self.comm is None:
            dist.all_gather_into_tensor(self._meta[:g], inputs.route_meta_local)
            meta = self._meta[:g]
        else:
            meta = self.comm.all_gather(
                inputs.route_meta_local, out=self._meta[:g]
            )
        w, i = self._w[:g], self._i[:g]
        i.copy_(meta[:, :k])
        w.view(torch.int32).copy_(meta[:, k:])
        return w, i


class MoeSortingStep:
    """Step 1: the expert sort, lifted out of ``fused_moe`` and never fused.

    Replicates the sorting block of ``_fused_moe_impl`` (fused_moe.py:1261-1328)
    rather than wrapping it, because the whole point is to run it as its own
    step.  Both branches are needed: the two GEMM1 families in the tuned CSVs
    want different sorts.  ``flydsl_mxmoe_g1_*`` asks for the Opus aux sort and
    gets back two extra tensors the port consumes; ``flydsl_moe1_*`` takes the
    ordinary sort and returns five.  The result is normalized to one 7-tuple so
    everything downstream is family-agnostic.
    """

    def __init__(self, weights: TpMoeWeights):
        self.shape = weights.shape

    def __call__(self, w_all, i_all, plan: _TunedPlan):
        metadata = plan.metadata
        if metadata.output_aux:
            return moe_sorting(
                i_all,
                w_all,
                self.shape.experts,
                self.shape.model_dim,
                dtypes.bf16,
                plan.block_m,
                # The atomic epilogue accumulates into this buffer, so the sort
                # has to zero it; the reduce epilogue owns its own intermediate.
                accumulate=plan.accumulate,
                output_aux=metadata.output_aux,
                output=None,
            )
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
            moe_sorting(
                i_all,
                w_all,
                self.shape.experts,
                self.shape.model_dim,
                dtypes.bf16,
                plan.block_m,
                accumulate=plan.accumulate,
                flat=metadata.flat,
                output=None,
            )
        )
        return (
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            None,  # m_indices        -- Opus-only
            None,  # reverse_sorted   -- Opus-only
        )


class QuantizeLocal:
    """Step 2: MXFP4-quantize this rank's own shard, before it goes on the wire.

    Per-1x32 MX quantization is row-local, so quantizing m rows here and
    gathering the result is bit-identical to gathering BF16 and quantizing all M
    rows -- and it moves 3.77x fewer bytes while doing 1/TP of the work.
    """

    def __init__(self):
        self._quant = get_hip_quant(QUANT_TYPE)

    def __call__(self, x_local: torch.Tensor):
        return self._quant(x_local, quant_dtype=AQ_DTYPE)


class PayloadAllGather:
    """Step 3: gather the quantized activation and its E8M0 scales.

    Two collectives on the buffers the quantizer already produced, and no
    staging at all.  The previous shape of this packed payload and scale into
    one uint8 row, gathered once, then called ``.contiguous()`` on both halves
    to get them back -- four extra kernels and two allocations to save one
    collective launch.  That trade is a loss once the collective is the
    one-shot kernel, which costs ~7.3us: the four staging copies cost more than
    the launch they save.  It also made the leg report ~65us at dsv4 M=8 for
    what measures 9.7us of device work here, which is most of why the split
    baseline looked far worse than a real unfused layer.

    On RCCL the two forms are close (both ~60-65us eager), so nothing is given
    up by choosing the one that has no staging in it.

    Both halves are plain bytes on the wire, so each rides whichever backend
    ``TpCollectives`` picks; MXFP4 and E8M0 are bit-cast for the one-shot
    kernel and cast straight back.
    """

    def __init__(
        self,
        shape: ModelShape,
        tp_size: int,
        max_local_tokens: int,
        device,
        comm: TpCollectives | None = None,
    ):
        total = max_local_tokens * tp_size
        self.tp_size = tp_size
        self.model_dim = shape.model_dim
        self.payload_bytes = shape.model_dim // 2
        self.scale_bytes = shape.model_dim // 32
        self.row_bytes = self.payload_bytes + self.scale_bytes
        self.comm = comm
        self._payload = torch.empty(
            (total, self.payload_bytes), dtype=torch.uint8, device=device
        )
        self._scale = torch.empty(
            (total, self.scale_bytes), dtype=torch.uint8, device=device
        )

    def wire_bytes(self, local_tokens: int) -> int:
        return local_tokens * self.row_bytes * (self.tp_size - 1)

    def __call__(self, xq, xq_scale, inputs: TpMoeInputs):
        m, g = inputs.local_tokens, inputs.global_tokens
        payload = xq.view(torch.uint8).view(m, self.payload_bytes)
        scale = xq_scale.view(torch.uint8).view(m, self.scale_bytes)
        if self.comm is None:
            dist.all_gather_into_tensor(self._payload[:g], payload)
            dist.all_gather_into_tensor(self._scale[:g], scale)
            a1, a1_scale = self._payload[:g], self._scale[:g]
        else:
            a1 = self.comm.all_gather(payload, out=self._payload[:g])
            a1_scale = self.comm.all_gather(scale, out=self._scale[:g])
        return a1.view(AQ_DTYPE), a1_scale.view(dtypes.fp8_e8m0)


class SplitLocalGemms:
    """Steps 4 and 5: GEMM1 and GEMM2 on an already-sorted, already-quantized input.

    ``fused_moe_2stages`` takes the sort results as plain arguments, so calling
    it directly is what "hoist the sort" means in practice -- no reimplementation
    of the GEMM dispatch, and the tuned rows stay exactly the ones
    ``test_moe_2stage.py`` benchmarks.  ``_metadata_transform`` pins the row this
    M resolved to so the internal lookup cannot pick a different one, and sets
    ``prequant`` so the FP4 activation is passed through with only its scale
    sorted instead of being quantized a second time (fused_moe.py:3767-3779).
    """

    def __init__(self, weights: TpMoeWeights):
        shape = weights.shape
        self.weights = weights
        self.shape = shape
        situ = shape.act_type == aiter.ActivationType.Situv2
        self.kwargs = {
            "activation": shape.act_type,
            "quant_type": QUANT_TYPE,
            "doweight_stage1": False,
            "q_dtype_a": AQ_DTYPE,
            "q_dtype_w": WQ_DTYPE,
            "w1_scale": weights.w1_scale,
            "w2_scale": weights.w2_scale,
            "hidden_pad": 0,
            "intermediate_pad": 0,
            "bias1": None,
            "bias2": None,
            "swiglu_limit": None,
            "beta": DEFAULT_SITUV2_BETA if situ else None,
            "linear_beta": DEFAULT_SITUV2_LINEAR_BETA if situ else None,
            "gate_mode": GateMode.SEPARATED.value,
            "routing_num_experts": shape.experts,
        }

    def __call__(self, a1, a1_scale, w_all, i_all, sorted_ret, plan: _TunedPlan):
        (
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            m_indices,
            reverse_sorted,
        ) = sorted_ret
        forced = plan.metadata
        if plan.prequant:
            forced = replace(forced, prequant=True)
        return fused_moe_2stages(
            a1,
            self.weights.w1,
            self.weights.w2,
            self.shape.topk,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            True,  # isG1U1
            plan.block_m,
            a1_scale=a1_scale,
            topk_ids=i_all,
            topk_weights=w_all,
            m_indices=m_indices,
            reverse_sorted=reverse_sorted,
            _metadata_transform=lambda _metadata: forced,
            **self.kwargs,
        )

    def capture_stage_callables(self, *args):
        """Run one eager pass and return ``({"stage1": call, "stage2": call}, out)``."""
        captured: list = []
        aiter.fused_moe.kernel_bench_callable = captured
        try:
            out = self(*args)
        finally:
            aiter.fused_moe.kernel_bench_callable = None
        return dict(captured), out


class SplitTpMoe:
    """sort -> quant -> AG -> GEMM1 -> GEMM2 -> RS, each step separately timed."""

    name = "split"

    def __init__(
        self,
        weights: TpMoeWeights,
        ctx: DistCtx,
        max_local_tokens: int,
        comm: TpCollectives | None = None,
    ):
        shape = weights.shape
        tp = weights.tp_size
        self.weights = weights
        self.shape = shape
        self.ctx = ctx
        self.tp_size = tp
        self.comm = comm
        self.plans = TunedPlans(weights)
        self.route_ag = RouteAllGather(shape, tp, max_local_tokens, ctx.device, comm)
        self.sorting = MoeSortingStep(weights)
        self.quant = QuantizeLocal()
        self.payload_ag = PayloadAllGather(
            shape, tp, max_local_tokens, ctx.device, comm
        )
        self.gemms = SplitLocalGemms(weights)
        self.reduce_scatter = ReduceScatterOutput(
            shape, tp, max_local_tokens, ctx.device, comm
        )
        # Only the accuracy path needs a BF16 view of the gathered activation.
        self.allgather = AllGatherTokens(
            shape, tp, max_local_tokens, ctx.device, comm
        )

    def kernel_names(self, global_tokens: int) -> tuple[str, str]:
        plan = self.plans(global_tokens)
        return plan.gemm1_kernel, plan.gemm2_kernel

    def steps(self, inputs: TpMoeInputs):
        """Run the chain once and hand back every intermediate, for timing."""
        plan = self.plans(inputs.global_tokens)
        w_all, i_all = self.route_ag(inputs)
        sorted_ret = self.sorting(w_all, i_all, plan)
        if plan.prequant:
            xq, xq_scale = self.quant(inputs.x_local)
            a1, a1_scale = self.payload_ag(xq, xq_scale, inputs)
        else:
            # The tuned GEMM1 at this M quantizes inline, so the wire stays BF16
            # and the quantize step is empty -- charging it separately here would
            # double-count work that happens inside GEMM1.
            xq = xq_scale = None
            a1, a1_scale = self.allgather(inputs)[0], None
        return plan, w_all, i_all, sorted_ret, xq, xq_scale, a1, a1_scale

    def __call__(self, inputs: TpMoeInputs) -> torch.Tensor:
        plan, w_all, i_all, sorted_ret, _xq, _s, a1, a1_scale = self.steps(inputs)
        partial = self.gemms(a1, a1_scale, w_all, i_all, sorted_ret, plan)
        return self.reduce_scatter(partial, inputs.local_tokens)

    def warmup(self, x, wts, ids) -> None:
        """Build every JIT module this M needs, single rank, no collectives."""
        plan = self.plans(int(x.shape[0]))
        sorted_ret = self.sorting(wts, ids, plan)
        if plan.prequant:
            a1, a1_scale = self.quant(x)
        else:
            a1, a1_scale = x, None
        self.gemms(a1, a1_scale, wts, ids, sorted_ret, plan)


# ---------------------------------------------------------------------------
# Placeholder for the fused kernel
# ---------------------------------------------------------------------------
class MegaMoeTpNotImplemented(NotImplementedError):
    pass


class MegaMoeTP:
    """Test-side adapter for the fused AG + GEMM1 + GEMM2 + RS MoE runtime.

    The engine lives in :mod:`aiter.ops.flydsl.mega_moe_tp`; this class only maps the
    test's ``TpMoeWeights`` / ``TpMoeInputs`` onto it, so ``MegaMoeTP`` stays
    interchangeable with :class:`SplitTpMoe`::

        moe = MegaMoeTP(weights, ctx, max_local_tokens=...)
        y_local = moe(inputs)          # [m, model_dim] bf16
    """

    name = "mega_moe_tp"

    IMPLEMENTED = True

    def __init__(
        self,
        weights: TpMoeWeights,
        ctx: DistCtx,
        max_local_tokens: int,
        *,
        ag_wire_quant: str = "auto",
        rs_wire_quant: str = "auto",
    ):
        self.weights = weights
        self.shape = weights.shape
        self.ctx = ctx
        self.tp_size = weights.tp_size
        self.max_local_tokens = max_local_tokens
        self.max_global_tokens = max_local_tokens * weights.tp_size
        self.ag_wire_quant = ag_wire_quant
        self.rs_wire_quant = rs_wire_quant
        situ = self.shape.act_type == aiter.ActivationType.Situv2
        self.config = MegaMoeTPConfig(
            rank=ctx.rank,
            world_size=weights.tp_size,
            model_dim=self.shape.model_dim,
            inter_dim=weights.local_inter_dim,
            experts=self.shape.experts,
            topk=self.shape.topk,
            max_local_tokens=max_local_tokens,
            activation=self.shape.act_type,
            beta=DEFAULT_SITUV2_BETA if situ else None,
            linear_beta=DEFAULT_SITUV2_LINEAR_BETA if situ else None,
            ag_wire=ag_wire_quant,
            rs_wire=rs_wire_quant,
        )
        self.engine = MegaMoeTPEngine(
            self.config,
            w1=weights.w1,
            w1_scale=weights.w1_scale,
            w2=weights.w2,
            w2_scale=weights.w2_scale,
            device=ctx.device,
        )

    @classmethod
    def is_available(cls, shape: ModelShape | None = None, tp_size: int = 8) -> bool:
        """Whether the fused kernel can serve this shape on this machine."""
        if not cls.IMPLEMENTED:
            return False
        return mega_moe_tp_supported()

    @classmethod
    def unavailable_reason(cls) -> str:
        if not cls.IMPLEMENTED:
            return "MegaMoeTP.IMPLEMENTED is False (fused kernel not written yet)"
        if not mega_moe_tp_supported():
            return f"fused TP MoE needs {SUPPORTED_GFX}, found {get_gfx()}"
        return ""

    def plan(self, global_tokens: int):
        return self.engine.plan(global_tokens)

    def __call__(self, inputs: TpMoeInputs) -> torch.Tensor:
        return self.engine(
            inputs.x_local,
            inputs.topk_weights_local,
            inputs.topk_ids_local,
        )


# ---------------------------------------------------------------------------
# Torch reference (same building blocks as test_moe_2stage.py)
# ---------------------------------------------------------------------------
class TorchReferenceTpMoe:
    """Reference for one TP rank's view of the layer.

    Mirrors ``test_moe_2stage.py``: quantize the activation, ``torch_moe_stage1``,
    requantize the intermediate, ``torch_moe_stage2``.  The only additions are
    the AllGather on the way in and, on the way out, an all-reduce of the
    per-rank partials followed by taking this rank's token shard -- which is
    exactly what the ReduceScatter computes.
    """

    def __init__(self, weights: TpMoeWeights, ctx: DistCtx):
        self.weights = weights
        self.shape = weights.shape
        self.ctx = ctx
        situ = weights.shape.act_type == aiter.ActivationType.Situv2
        self.situ_beta = DEFAULT_SITUV2_BETA if situ else 1.0
        self.situ_linear_beta = DEFAULT_SITUV2_LINEAR_BETA if situ else 1.0

    @torch.no_grad()
    def __call__(self, inputs: TpMoeInputs, allgather: AllGatherTokens):
        shape = self.shape
        w = self.weights
        x_all, w_all, i_all = allgather(inputs)
        x_all, w_all, i_all = x_all.clone(), w_all.clone(), i_all.clone()

        # a4w4 activation quant: test_moe_2stage.py:274 (the generic else branch,
        # since AQDType=fp4x2 is not in [bf16, fp16, fp8]).
        torch_quant = aiter.get_torch_quant(QUANT_TYPE)
        a1_qt, a1_scale = torch_quant(x_all, quant_dtype=AQ_DTYPE)

        out1 = torch_moe_stage1(
            a1_qt,
            w.w1_ref,
            w.w2_ref,
            w_all,
            i_all,
            dtype=dtypes.bf16,
            activation=shape.act_type,
            quant_type=QUANT_TYPE,
            a1_scale=a1_scale,
            w1_scale=w.w1_scale_ref,
            w1_bias=None,
            doweight=False,
            swiglu_limit=None,
            situ_beta=self.situ_beta,
            situ_linear_beta=self.situ_linear_beta,
        )

        a2_qt, a2_scale = torch_quant(out1, quant_dtype=AQ_DTYPE)
        a2_qt = a2_qt.view(inputs.global_tokens, shape.topk, -1)

        partial = torch_moe_stage2(
            a2_qt,
            w.w1_ref,
            w.w2_ref,
            w_all,
            i_all,
            dtype=dtypes.bf16,
            quant_type=QUANT_TYPE,
            w2_scale=w.w2_scale_ref,
            a2_scale=a2_scale,
            w2_bias=None,
            doweight=True,
        )

        # ReduceScatter == all-reduce then slice this rank's shard.
        acc = partial.float()
        dist.all_reduce(acc)
        m = inputs.local_tokens
        start = self.ctx.rank * m
        return acc[start : start + m]


def rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Global (cross-rank) relative L2 error."""
    err = torch.sum((actual.float() - expected.float()) ** 2)
    ref = torch.sum(expected.float() ** 2)
    dist.all_reduce(err)
    dist.all_reduce(ref)
    if float(ref.item()) == 0.0:
        return float("nan")
    return float(torch.sqrt(err / ref).item())


# ---------------------------------------------------------------------------
# One (model, tokens) case
# ---------------------------------------------------------------------------
class SkipCase(Exception):
    """Raised when every rank has already agreed to skip the current case."""


def jit_warmup(
    moe: SplitTpMoe, shape: ModelShape, ctx: DistCtx, global_tokens: int
) -> bool:
    """Build every JIT module on rank 0 alone, then let the others in.

    Two problems this avoids:

    * Some shapes resolve to a CK 2-stage kernel whose module is not prebuilt.
      When all eight ranks reach it together they race on the same build and the
      ranks that lose the file lock die with ``ModuleNotFoundError`` on the .so
      that is still being written.
    * Some of those modules do not build at all. On ROCm 7.1.0 the a4w4 CK
      stage2 module fails with "non-type template argument is not a constant
      expression" inside
      ``ck/.../blockwise_gemm_pipeline_xdlops_b_preshuffle_mx_moe_gufusion_v3.hpp``,
      so e.g. dsv3 at M>=128 cannot run here at all. Detecting that on one rank
      lets the whole group skip the case instead of aborting the sweep.

    The local MoE involves no collectives, so rank 0 can compile it while the
    others wait; the jit cache is on the shared filesystem, so they then find it
    ready. Returns whether every rank should proceed.
    """
    ok = True
    if ctx.is_main:
        try:
            x = torch.randn(
                (global_tokens, shape.model_dim), dtype=dtypes.bf16, device=ctx.device
            )
            ids = (
                torch.arange(
                    global_tokens * shape.topk, dtype=torch.int32, device=ctx.device
                )
                % shape.experts
            ).view(global_tokens, shape.topk)
            wts = torch.full(
                (global_tokens, shape.topk),
                1.0 / shape.topk,
                dtype=torch.float32,
                device=ctx.device,
            )
            moe.warmup(x, wts, ids)
            del x, ids, wts
        except Exception as exc:  # noqa: BLE001 - warmup must never be fatal
            logger.warning("[jit-warmup] %s M=%d: %s", shape.name, global_tokens, exc)
            ok = False
        torch.cuda.empty_cache()
    return all_ranks_ok(ok, ctx.device)



def _graph_capture_ctx():
    """aiter's capture context, or a no-op when its parallel state is absent.

    ``CustomAllreduce`` needs to know it is being captured: under capture it
    routes through the pre-registered IPC pool instead of the normal path, and
    capturing it without this yields a graph that replays stale peer pointers.
    The fused engine does not need it, but the split baseline does as soon as
    its collectives are the one-shot kernels, and both go through this helper
    so the two sides are captured identically.
    """
    try:
        from aiter.dist.parallel_state import get_tp_group, graph_capture

        get_tp_group()
    except Exception:  # noqa: BLE001 - no aiter parallel state -> plain capture
        import contextlib

        return contextlib.nullcontext()
    return graph_capture()


def _device_us(fn, *, reps: int, rounds: int, device) -> float:
    """Device time for one call to ``fn``, with host dispatch removed.

    Neither of the obvious measurements can see a leg this small.  Timing an
    eager loop charges ~11-35us of host dispatch per call depending on the
    backend, which at decode sizes is larger than the work.  Replaying a graph
    that holds a *single* call is no better: the replay call itself costs ~14us
    on this box, so a one-call graph reports ~14us for a no-op.  Capturing
    ``reps`` copies into one graph and replaying that amortizes the replay to
    ``1/reps`` and leaves device work.

    This is how the legs were shown to be mostly dispatch. dsv4 M=8, TP8, the
    one-shot backend, eager vs device::

        route AG  58.2 -> 7.5     AllGather 101.6 -> 9.7
        sort      41.2 -> 9.7     ReduceSc.  47.7 -> 6.0
        quant     20.6 -> 1.9

    Returns NaN when capture is not possible -- a diagnostic column must not
    take the sweep down with it.
    """
    graph = None
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        barrier()
        graph = torch.cuda.CUDAGraph()
        with _graph_capture_ctx(), torch.cuda.graph(graph):
            for _ in range(reps):
                fn()
        torch.cuda.synchronize()
        mean, _ = time_us(
            graph.replay, iters=4, warmup=2, device=device, rounds=rounds
        )
        return mean / reps
    except Exception as exc:  # noqa: BLE001 - diagnostic column only
        logger.warning("[TP-MOE] leg capture failed: %s", exc)
        return float("nan")
    finally:
        del graph


def _time_graph(impl, inputs, args, ctx) -> tuple[float, torch.Tensor | None]:
    """Capture one forward into a CUDA graph and time the replay.

    Works for either implementation. Replay removes *all* host-side dispatch,
    which is the only way the two sides can be compared the way a serving stack
    actually runs them -- a decode layer is captured, not dispatched per op, so
    an eager-vs-eager speedup credits the fused kernel for launch overhead that
    production never pays. See ``--graph``.

    Returns NaN rather than raising when capture is not possible: this is a
    diagnostic column, and a shape that cannot be captured should not take the
    sweep down with it.
    """
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                impl(inputs)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        barrier()
        graph = torch.cuda.CUDAGraph()
        with _graph_capture_ctx(), torch.cuda.graph(graph):
            out = impl(inputs)
        torch.cuda.synchronize()
        mean, _ = time_us(
            graph.replay,
            iters=args.iters,
            warmup=args.warmup,
            device=ctx.device,
            rounds=args.rounds,
        )
        # A replay that is fast but wrong is worthless, and every fused kernel
        # here takes its epoch from the arena rather than the host precisely so
        # replay stays correct -- so check it rather than assume it.
        graph.replay()
        torch.cuda.synchronize()
        replayed = out.clone()
        return mean, replayed
    except Exception as exc:  # noqa: BLE001 - diagnostic column only
        logging.getLogger(__name__).warning("[TP-MOE] graph capture failed: %s", exc)
        return float("nan"), None


def run_case(
    shape: ModelShape,
    weights: TpMoeWeights,
    ctx: DistCtx,
    args,
    global_tokens: int,
    max_local_tokens: int,
) -> dict:
    tp = args.tp
    inputs = make_inputs(shape, ctx, tp, global_tokens, args.seed, args.route)
    moe = SplitTpMoe(weights, ctx, max_local_tokens, comm=args.comm_backend)
    kname1, kname2 = moe.kernel_names(global_tokens)
    if not args.no_jit_warmup and not jit_warmup(moe, shape, ctx, global_tokens):
        raise SkipCase(
            "kernel build failed on rank 0 (see the [jit-warmup] warning above)"
        )

    row: dict = {
        "model": shape.name,
        "tp": tp,
        "global_tokens": global_tokens,
        "local_tokens": inputs.local_tokens,
        "model_dim": shape.model_dim,
        "inter_dim_local": weights.local_inter_dim,
        "experts": shape.experts,
        "topk": shape.topk,
        "act": str(shape.act_type).split(".")[-1],
        "comm": args.comm_backend.describe(),
        "gemm1_kernel": kname1,
        "gemm2_kernel": kname2,
        "rel_l2": float("nan"),
    }

    # ---------------- accuracy ----------------
    y = moe(inputs)
    y_actual = y.clone()
    barrier()
    if global_tokens <= args.accuracy_max_tokens:
        reference = TorchReferenceTpMoe(weights, ctx)
        y_expected = reference(inputs, moe.allgather)
        row["rel_l2"] = rel_l2(y_actual, y_expected)
        del y_expected
        torch.cuda.empty_cache()
        if not (row["rel_l2"] == row["rel_l2"]):  # NaN
            raise AssertionError(f"{shape.tag(tp)} tokens={global_tokens}: rel_l2 NaN")
        if row["rel_l2"] >= args.rtol:
            raise AssertionError(
                f"{shape.tag(tp)} tokens={global_tokens}: "
                f"rel_l2={row['rel_l2']:.6f} exceeds {args.rtol}"
            )
    has_nan = bool(torch.isnan(y_actual).any().item())
    nan_flag = torch.tensor(int(has_nan), dtype=torch.int32, device=ctx.device)
    dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)
    if int(nan_flag.item()):
        raise AssertionError(f"{shape.tag(tp)} tokens={global_tokens}: output has NaN")

    # ---------------- fused implementation: accuracy ----------------
    # Run before the early `--no-perf` return so an accuracy-only sweep still
    # exercises the fused path. `fused` stays alive for the timing block below.
    fused = None
    if args.impl in ("fused", "both"):
        if not MegaMoeTP.is_available(shape, tp):
            row["fused_us"] = float("nan")
            row["fused_note"] = MegaMoeTP.unavailable_reason()
        else:
            fused = MegaMoeTP(
                weights,
                ctx,
                max_local_tokens,
                ag_wire_quant=args.ag_wire,
                rs_wire_quant=args.rs_wire,
            )
            plan = fused.plan(global_tokens)
            row["fused_ag_wire"] = plan.ag_wire
            # Non-zero only when this M's own tuned row quantizes inline and a
            # larger bucket's row was borrowed to stay on the FP4 wire, which
            # means the GEMMs below are not the ones the tuner picked for this M.
            row["fused_gemm_bucket"] = (
                plan.gemm_bucket if plan.gemm_bucket != plan.tokens else 0
            )
            y_fused = fused(inputs)
            # Normally the unfused path is the oracle: both run the same tuned
            # GEMMs, so anything beyond quantization noise is a fusion bug, and
            # the tight fused_rtol applies.  When a row was borrowed the two
            # sides run *different* kernels, so their difference is ordinary
            # kernel-to-kernel rounding and only the wider reference tolerance
            # is meaningful; torch stays the oracle via fused_ref_rel_l2 below.
            borrowed_row = bool(row["fused_gemm_bucket"])
            fused_gate = args.rtol if borrowed_row else args.fused_rtol
            row["fused_rel_l2"] = rel_l2(y_fused, y_actual)
            del y_fused
            if not (row["fused_rel_l2"] == row["fused_rel_l2"]):  # NaN
                raise AssertionError(
                    f"{shape.tag(tp)} tokens={global_tokens}: fused rel_l2 NaN"
                )
            if row["fused_rel_l2"] >= fused_gate:
                why = (
                    f" (borrowed the M={row['fused_gemm_bucket']} tuned row)"
                    if borrowed_row
                    else ""
                )
                raise AssertionError(
                    f"{shape.tag(tp)} tokens={global_tokens}: fused vs unfused "
                    f"rel_l2={row['fused_rel_l2']:.6f} exceeds {fused_gate}{why}"
                )
            if global_tokens <= args.accuracy_max_tokens:
                reference = TorchReferenceTpMoe(weights, ctx)
                y_expected = reference(inputs, moe.allgather)
                row["fused_ref_rel_l2"] = rel_l2(fused(inputs), y_expected)
                del y_expected
                torch.cuda.empty_cache()
                if row["fused_ref_rel_l2"] >= args.rtol:
                    raise AssertionError(
                        f"{shape.tag(tp)} tokens={global_tokens}: fused vs torch "
                        f"rel_l2={row['fused_ref_rel_l2']:.6f} exceeds {args.rtol}"
                    )

    if args.no_perf:
        return row

    # ---------------- per-module perf ----------------
    timer = functools.partial(
        time_us,
        iters=args.iters,
        warmup=args.warmup,
        device=ctx.device,
        rounds=args.rounds,
    )
    # Same order the fused kernel will run: the sort is hoisted out in front and
    # everything after it is what gets collapsed into one launch.
    quant_mean = 0.0
    plan, w_all, i_all, sorted_ret, xq, xq_scale, a1, a1_scale = moe.steps(inputs)
    stages, _ = moe.gemms.capture_stage_callables(
        a1, a1_scale, w_all, i_all, sorted_ret, plan
    )
    partial = moe.gemms(a1, a1_scale, w_all, i_all, sorted_ret, plan)

    route_mean, _ = timer(lambda: moe.route_ag(inputs))
    sort_mean, _ = timer(lambda: moe.sorting(w_all, i_all, plan))
    if plan.prequant:
        quant_mean, _ = timer(lambda: moe.quant(inputs.x_local))
        ag_mean, ag_max = timer(lambda: moe.payload_ag(xq, xq_scale, inputs))
        ag_bytes = moe.payload_ag.wire_bytes(inputs.local_tokens)
    else:
        # GEMM1 quantizes inline at this M, so there is no separate quantize to
        # charge and the wire stays BF16.
        ag_mean, ag_max = timer(lambda: moe.allgather(inputs))
        ag_bytes = moe.allgather.wire_bytes(inputs.local_tokens)
    moe_mean, _ = timer(lambda: moe.gemms(a1, a1_scale, w_all, i_all, sorted_ret, plan))
    row["ag_wire"] = plan.ag_wire

    g1_mean = g2_mean = float("nan")
    if "stage1" in stages:
        g1_mean, _ = timer(stages["stage1"])
    if "stage2" in stages:
        g2_mean, _ = timer(stages["stage2"])
    rs_mean, rs_max = timer(lambda: moe.reduce_scatter(partial, inputs.local_tokens))
    e2e_mean, e2e_max = timer(lambda: moe(inputs))

    # What one fused launch is supposed to replace.  The sort and the routing
    # AllGather stay outside it, so they are excluded -- this is the number the
    # fused column has to beat, not e2e.
    fusable_mean = quant_mean + ag_mean + moe_mean + rs_mean

    row.update(
        {
            "route_ag_us": route_mean,
            "sort_us": sort_mean,
            "quant_us": quant_mean,
            "ag_us": ag_mean,
            "gemm1_us": g1_mean,
            "gemm2_us": g2_mean,
            "moe_us": moe_mean,
            # Scale sort + top-k reduction: inside the GEMM block but not in
            # either GEMM.
            "moe_other_us": moe_mean - (g1_mean + g2_mean),
            "rs_us": rs_mean,
            "fusable_us": fusable_mean,
            "e2e_us": e2e_mean,
            "e2e_max_us": e2e_max,
            # e2e is eager, so it also carries the host-side dispatch cost of
            # chaining the modules (tuned-config lookup, moe_sorting wrapper,
            # per-launch Python). At small M that is a real part of the gap the
            # fused kernel closes, so keep it visible rather than hiding it.
            #
            # It is a *residual*, not a measurement, and a biased one: every leg
            # takes the minimum over its own rounds independently of e2e, so a
            # sum of independent minima is <= the minimum of the sum even on an
            # idle node. A large value therefore means either real dispatch cost
            # or that one timed region caught interference the others missed --
            # see host_gap_pct, which flags the rows worth re-measuring.
            "host_gap_us": e2e_mean - (route_mean + sort_mean + fusable_mean),
            "host_gap_pct": (
                100.0
                * (e2e_mean - (route_mean + sort_mean + fusable_mean))
                / e2e_mean
                if e2e_mean
                else 0.0
            ),
            "comm_pct": (
                100.0 * (route_mean + ag_mean + rs_mean) / e2e_mean
                if e2e_mean
                else 0.0
            ),
            "ag_GBps": ag_bytes / ag_max / 1e3,
            "rs_GBps": moe.reduce_scatter.wire_bytes(inputs.local_tokens)
            / rs_max
            / 1e3,
        }
    )

    if args.leg_device_time:
        # One leg per run, because one capture per process is the only thing
        # this stack survives reliably -- see --leg-device-time. This is the
        # answer to "what does the AllGather actually cost": the eager columns
        # above are dominated by per-launch dispatch at decode sizes and say
        # more about the harness than about the layer.
        legs = {
            "route_ag": lambda: moe.route_ag(inputs),
            "sort": lambda: moe.sorting(w_all, i_all, plan),
            "rs": lambda: moe.reduce_scatter(partial, inputs.local_tokens),
            "ag": (
                (lambda: moe.payload_ag(xq, xq_scale, inputs))
                if plan.prequant
                else (lambda: moe.allgather(inputs))
            ),
        }
        if plan.prequant:
            legs["quant"] = lambda: moe.quant(inputs.x_local)
        leg_fn = legs.get(args.leg_device_time)
        if leg_fn is None:
            # 'quant' on a bf16-wire bucket: GEMM1 quantizes inline, so there
            # is no standalone quantize to time. Report it as absent, not zero.
            row[f"{args.leg_device_time}_dev_us"] = float("nan")
        else:
            row[f"{args.leg_device_time}_dev_us"] = _device_us(
                leg_fn, reps=args.leg_reps, rounds=args.rounds, device=ctx.device
            )

    if args.graph:
        # Capture the *baseline* too. Without this the graph column compares a
        # replayed fused path against an eager control group, so every launch
        # the baseline dispatches counts as fusion win -- which at M<=32 is a
        # third of the reported gap and is not a gap production has, because a
        # decode layer is captured on both sides.
        row["split_graph_us"], replayed = _time_graph(moe, inputs, args, ctx)
        # A replay that is fast but wrong would make the baseline look
        # arbitrarily good, and the baseline is the whole claim here -- so gate
        # it the same way the fused replay is gated rather than assume the
        # capture was faithful. The collectives are the risk: under capture
        # CustomAllreduce takes a different path through its registered pool.
        if replayed is not None:
            row["split_graph_rel_l2"] = rel_l2(replayed, y_actual)
            del replayed
            if row["split_graph_rel_l2"] >= args.rtol:
                raise AssertionError(
                    f"{shape.tag(tp)} tokens={global_tokens}: split graph "
                    f"replay rel_l2={row['split_graph_rel_l2']:.6f} exceeds "
                    f"{args.rtol} -- the captured baseline is not computing "
                    "the same thing as the eager one, so split_graph_us is "
                    "meaningless"
                )

    if args.probe_quant_wire:
        probe = QuantWireAllGatherProbe(shape, tp, max_local_tokens, ctx.device)
        probe(inputs)
        agq_mean, _ = timer(lambda: probe(inputs))
        quant_mean, _ = timer(lambda: probe.quantize_local(inputs))
        row["ag_fp4_us"] = agq_mean
        row["ag_speedup"] = ag_mean / agq_mean if agq_mean else float("nan")
        row["quant_local_us"] = quant_mean
        row["ag_bytes_ratio"] = moe.allgather.wire_bytes(
            inputs.local_tokens
        ) / probe.wire_bytes(inputs.local_tokens)

    # ---------------- fused implementation, when it exists ----------------
    if fused is not None:
        engine = fused.engine
        plan = engine.plan(global_tokens)
        # Break the fused path down the same way as the unfused one, so the two
        # columns are comparable leg by leg rather than only end to end.
        f_ag_mean, _ = timer(
            lambda: engine.all_gather(
                inputs.x_local,
                inputs.topk_weights_local,
                inputs.topk_ids_local,
                plan=plan,
            )
        )
        a1, a1_scale, f_w, f_i = engine.all_gather(
            inputs.x_local, inputs.topk_weights_local, inputs.topk_ids_local, plan=plan
        )
        # Same local MoE, driven by the unfused wrapper on the *fused* inputs.
        # Isolates "the arena buffers / prequantized operand made the GEMMs
        # slower" from "the fused driver's own dispatch costs more".
        f_moe_mean, _ = timer(lambda: engine.local_moe(a1, a1_scale, f_w, f_i, plan))
        if plan.fuse_rs:
            # The ReduceScatter rides in GEMM2's tail, so there is no separate
            # leg to time: what is comparable to (unfused GEMM + RS) is the one
            # call that now does both. Report it as the RS leg's cost *over* the
            # bare MoE so the two columns still add up to the same total.
            f_moe_rs_mean, _ = timer(
                lambda: engine.local_moe(
                    a1, a1_scale, f_w, f_i, plan, inputs.local_tokens
                )
            )
            f_rs_mean = max(0.0, f_moe_rs_mean - f_moe_mean)
        else:
            f_rs_mean, _ = timer(
                lambda: engine.reduce_scatter(inputs.local_tokens, plan)
            )
        fused_mean, _ = timer(lambda: fused(inputs))
        if args.graph:
            # Everything on the fused path is a device-side-epoch kernel, so a
            # captured graph replays correctly. Replay removes *all* host-side
            # dispatch, so the gap between this and fused_us is exactly what
            # collapsing the remaining launches could still win -- and what is
            # left is real device work.
            row["fused_graph_us"], replayed = _time_graph(fused, inputs, args, ctx)
            if replayed is not None:
                row["fused_graph_rel_l2"] = rel_l2(replayed, fused(inputs))
                del replayed
            # The only apples-to-apples speedup: both sides replayed, neither
            # charged host dispatch.
            split_graph = row.get("split_graph_us", float("nan"))
            fused_graph = row.get("fused_graph_us", float("nan"))
            # `fused_graph > 0` is also the NaN guard: a failed capture returns
            # NaN, and NaN fails every comparison.
            row["graph_speedup"] = (
                split_graph / fused_graph if fused_graph > 0 else float("nan")
            )
        row.update(
            {
                "fused_ag_us": f_ag_mean,
                "fused_moe_us": f_moe_mean,
                "fused_rs_us": f_rs_mean,
                "fused_rs_mode": "fused" if plan.fuse_rs else "split",
                "stage12": engine.stage12_mode(plan, inputs.local_tokens),
                "fused_us": fused_mean,
                "fused_speedup": (
                    e2e_mean / fused_mean if fused_mean else float("nan")
                ),
                "ag_gain": ag_mean / f_ag_mean if f_ag_mean else float("nan"),
                "rs_gain": rs_mean / f_rs_mean if f_rs_mean else float("nan"),
            }
        )

    # Everything above is function-local, so it is released on return; the
    # caller calls empty_cache() between cases.  Do not `del` the tensors the
    # timing lambdas captured.
    return row


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
#: Above this global token count, host dispatch is a fixed cost that should be
#: a small share of e2e, so a large ``host_gap_pct`` there means a timed region
#: caught interference rather than real dispatch work.
_SUSPECT_MIN_TOKENS = 2048
#: Clean TP8 rows at those sizes measured 0.5-2.5%; the two contaminated ones
#: measured 10.0% and 11.4%. 8% separates them with room to spare.
_SUSPECT_GAP_PCT = 8.0


def _suspect_note(row: dict, global_tokens: int) -> str:
    """Flag a row whose leg accounting does not add up.

    ``host_gap_us`` is a residual, so interference in any one timed region shows
    up there rather than in the leg that actually suffered. Saying so at print
    time turns a silently wrong speedup into a visible one -- the alternative is
    reading a contaminated number as a result, which has already happened once.
    """
    gap_pct = row.get("host_gap_pct")
    if (
        gap_pct is None
        or global_tokens < _SUSPECT_MIN_TOKENS
        or gap_pct < _SUSPECT_GAP_PCT
    ):
        return ""
    return (
        f"  [SUSPECT host_gap={gap_pct:.0f}% of e2e; re-measure with more "
        "--rounds before trusting this row]"
    )


#: How much a smaller M may exceed the next larger M before it is called out.
#: Not pure noise slack: at TP8 the smallest sweep point is one local token,
#: whose fused path is reproducibly 6-8% slower than two local tokens on all
#: four models (measured at --rounds 15, so it is real behaviour and not
#: interference). The threshold sits above that floor; the contaminated rows
#: this check exists to catch measured 1.20x and 1.42x.
_MONOTONIC_SLACK = 1.12


def _report_non_monotonic(df) -> None:
    """Warn where a smaller token count timed slower than a larger one.

    The per-row ``host_gap`` check cannot fire at small M, where a large gap is
    honest dispatch cost. What still holds there is monotonicity: doubling the
    tokens cannot make the layer faster, so ``e2e(M) > e2e(2M)`` means the
    smaller cell caught interference. This is how kimi3 M=8 was caught reading
    1.49x when a clean re-measure gave 1.15x.
    """
    hits = []
    for col in ("e2e_us", "fused_us"):
        if col not in df.columns:
            continue
        for model, grp in df.groupby("model"):
            grp = grp.sort_values("global_tokens")
            toks = grp["global_tokens"].tolist()
            vals = grp[col].tolist()
            for i in range(len(vals) - 1):
                a, b = vals[i], vals[i + 1]
                if a == a and b == b and a > b * _MONOTONIC_SLACK:
                    hits.append((model, col, toks[i], a, toks[i + 1], b))
    if not hits:
        return
    print("\n[SUSPECT] smaller M timed slower than the next larger M:")
    for model, col, m0, v0, m1, v1 in hits:
        print(
            f"  {model} {col}: M={m0} {v0:.1f}us > M={m1} {v1:.1f}us "
            f"({v0 / v1:.2f}x) -- re-measure M={m0} with more --rounds"
        )


_PERF_COLUMNS = [
    "model",
    "global_tokens",
    "local_tokens",
    "inter_dim_local",
    "comm",
    "rel_l2",
    # Fusion order: the sort and the routing AllGather stay outside the fused
    # kernel; everything from quant to rs is what one launch replaces.
    "ag_wire",
    "route_ag_us",
    "route_ag_dev_us",
    "sort_us",
    "sort_dev_us",
    "quant_us",
    "quant_dev_us",
    "ag_us",
    "ag_dev_us",
    "gemm1_us",
    "gemm2_us",
    "moe_other_us",
    "rs_us",
    "rs_dev_us",
    "fusable_us",
    "host_gap_us",
    "host_gap_pct",
    "e2e_us",
    "split_graph_us",
    "split_graph_rel_l2",
    "comm_pct",
    "ag_GBps",
    "rs_GBps",
    # only present with --probe-quant-wire / --impl fused|both
    "ag_fp4_us",
    "ag_speedup",
    "ag_bytes_ratio",
    "quant_local_us",
    "fused_ag_wire",
    "fused_gemm_bucket",
    "fused_rel_l2",
    "fused_ref_rel_l2",
    "fused_ag_us",
    "fused_moe_us",
    "fused_rs_us",
    "fused_rs_mode",
    "stage12",
    "fused_graph_us",
    "fused_graph_rel_l2",
    "fused_us",
    "ag_gain",
    "rs_gain",
    "fused_speedup",
    "graph_speedup",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split("Usage")[0],
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=list(MODELS),
        choices=list(MODELS),
        help="Which model shapes to sweep.",
    )
    p.add_argument(
        "--tokens",
        type=int,
        nargs="*",
        default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        help="GLOBAL token counts (must be divisible by --tp). The default is the "
        "full decode-to-prefill sweep; every value is a power of two so the tuned "
        "CSV lookup (get_padded_M = nextPow2) hits an exact row. Trim it on a "
        "busy node -- the largest sizes need several GiB of free HBM per GPU.",
    )
    p.add_argument("--tp", type=int, default=0, help="TP size; 0 = WORLD_SIZE.")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Repeat each timed region this many times and keep the fastest "
        "round. Rejects interference from other tenants on a shared node.",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--accuracy-max-tokens",
        type=int,
        default=128,
        help="Skip the torch reference above this GLOBAL token count "
        "(it loops over every expert and gets slow fast).",
    )
    p.add_argument("--rtol", type=float, default=0.06, help="rel_l2 accuracy gate.")
    p.add_argument(
        "--fused-rtol",
        type=float,
        default=0.02,
        help="rel_l2 gate for fused-vs-unfused. Normally both run the same tuned "
        "GEMMs, so the only legitimate difference is where the activation "
        "quantization happens; anything larger is a fusion bug, hence the "
        "tighter bound. Cases that had to borrow another bucket's tuned row "
        "(fused_gemm_bucket != 0) run different kernels on the two sides and "
        "are held to --rtol instead.",
    )
    p.add_argument(
        "--ag-wire",
        choices=["auto", "fp4_1x32", "bf16"],
        default="auto",
        help="Wire format for the fused AllGather. 'auto' prefers the MXFP4 "
        "wire at every M, borrowing a larger bucket's tuned row when this M's "
        "own GEMM1 quantizes inline; 'bf16' pins the BF16 wire and always keeps "
        "the tuned row.",
    )
    p.add_argument(
        "--rs-wire",
        choices=["auto", "bf16"],
        default="auto",
        help="Wire format for the fused ReduceScatter.",
    )
    p.add_argument(
        "--comm",
        choices=["auto", "custom", "rccl"],
        default="auto",
        help="Collective backend for the SPLIT baseline. 'custom' uses aiter's "
        "one-shot P2P AllGather/ReduceScatter wherever the shape is eligible "
        "and falls back to RCCL where it is not; 'rccl' pins "
        "torch.distributed. Neither wins outright: the one-shot kernels are "
        "~2.3x faster on device at decode sizes (8us vs 18us at M=8) but carry "
        "2-3x the host dispatch, so RCCL is ahead in eager and the one-shot "
        "path is ahead under graph replay. 'auto' therefore picks by regime -- "
        "custom with --graph, RCCL without -- so the control group is the best "
        "available implementation of whatever is being measured rather than a "
        "fixed choice that happens to lose.",
    )
    p.add_argument(
        "--route",
        choices=["balanced", "random"],
        default="balanced",
        help="Expert routing distribution.",
    )
    p.add_argument("--no-perf", action="store_true", help="Accuracy only.")
    p.add_argument(
        "--no-jit-warmup",
        action="store_true",
        help="Skip the rank-0-only JIT warmup. Only safe once every kernel this "
        "sweep touches is already built, otherwise the ranks race on the build.",
    )
    p.add_argument(
        "--probe-quant-wire",
        action="store_true",
        help="Also measure an MXFP4 AllGather to size the quantized-wire win.",
    )
    p.add_argument(
        "--impl",
        choices=["unfused", "fused", "both"],
        default="unfused",
        help="Which implementations to run.",
    )
    p.add_argument(
        "--leg-device-time",
        choices=["route_ag", "sort", "quant", "ag", "rs"],
        default=None,
        help="Report ONE non-GEMM leg's DEVICE time (<leg>_dev_us) by "
        "capturing --leg-reps copies of it into a CUDA graph. The eager leg "
        "columns carry ~11-35us of host dispatch per launch, which at decode "
        "sizes exceeds the work being timed; this is what the leg actually "
        "costs -- at dsv4 M=8 the AllGather measures 9.7us here against 63-102us "
        "eager depending on backend. One leg per run on purpose: capturing all "
        "five in one process "
        "SIGSEGVs a rank about half the time on this stack (on both "
        "collective backends), and in-graph event records, which would have "
        "allowed a single capture, are unsupported here.",
    )
    p.add_argument(
        "--leg-reps",
        type=int,
        default=32,
        help="Copies of each leg captured into one graph for --leg-device-time. "
        "Amortizes the ~14us per-replay dispatch to 1/N of it.",
    )
    p.add_argument(
        "--graph",
        action=argparse.BooleanOptionalAction,
        # None means "not asked for either way" -> defaults to on, but a
        # multi-cell sweep is refused rather than driven into a SIGSEGV.
        # Passing --graph explicitly is taken as accepting that risk.
        default=None,
        help="Time BOTH paths replayed from a CUDA graph and report "
        "graph_speedup. ON BY DEFAULT: a decode layer is captured in "
        "production, not dispatched per op, so this is the regime the layer "
        "actually runs in -- and at small M it is not a small correction, it "
        "reverses the result (dsv4 M=8 reads 1.63x eager and 0.46x captured). "
        "The eager columns are still collected alongside. Two captures per "
        "cell, and repeated capture in one process kills a rank after a few "
        "cells (kimi3 died on the 4th), so by default a multi-cell run is "
        "refused with the per-cell loop to run instead; passing --graph "
        "explicitly overrides that. Use --no-graph for the old eager-only "
        "sweep, which does every cell in one process.",
    )
    p.add_argument("--csv", default=None, help="Write the perf table here (rank 0).")
    args = p.parse_args(argv)
    # --graph defaults on but records whether that was the caller's decision;
    # see the guard in main().
    args.graph_explicit = args.graph is not None
    if args.graph is None:
        args.graph = True
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # 'auto' resolves by measurement regime -- see --comm. Resolving it here,
    # before the process group comes up, also means an eager sweep never pays
    # the aiter parallel-state bring-up it would not use.
    resolved_comm = args.comm
    if resolved_comm == "auto":
        resolved_comm = "custom" if args.graph else "rccl"
    ctx = setup_dist(want_custom_comm=resolved_comm != "rccl")
    args.tp = args.tp or ctx.world
    args.comm_backend = TpCollectives(resolved_comm, ctx, args.tp)
    if args.comm == "custom" and not args.comm_backend.enabled:
        raise RuntimeError(
            "--comm custom requested but aiter's one-shot collectives are "
            f"unavailable: {args.comm_backend.error}"
        )
    try:
        if args.tp != ctx.world:
            raise ValueError(
                f"--tp={args.tp} must equal WORLD_SIZE={ctx.world} "
                "(this test uses the whole world as one TP group)"
            )
        gfx = get_gfx()
        if gfx not in SUPPORTED_GFX:
            if ctx.is_main:
                print(f"[SKIP] a4w4 MoE needs {SUPPORTED_GFX}, found {gfx}")
            return 0
        if os.environ.get("AITER_CONFIG_FMOE") and ctx.is_main:
            print(
                "[WARN] AITER_CONFIG_FMOE is set, which pins tuned-kernel lookup to a "
                "single CSV. Unset it so all four model shapes resolve to tuned "
                "kernels in one process.",
                flush=True,
            )
        captures_per_cell = (2 if args.graph else 0) + (
            1 if args.leg_device_time else 0
        )
        cells = len(args.models) * len(set(args.tokens))
        if args.graph and not args.graph_explicit and cells > 1:
            # --graph is on by default, and a default must not walk a long
            # sweep into a SIGSEGV. Capturing repeatedly in one process kills a
            # rank after a handful of cells (kimi3 on the 4th, dsv4 survived
            # 4), so the sweep has to be split. Refuse with the loop rather
            # than silently dropping to eager, which would quietly change the
            # regime the numbers came from.
            tok = " ".join(str(t) for t in sorted(set(args.tokens)))
            mods = " ".join(args.models)
            raise SystemExit(
                f"--graph is on by default and this run asks for {cells} "
                "cells; repeated CUDA graph capture in one process kills a "
                "rank after a few of them. Run one cell per process:\n\n"
                f"  for M in {mods}; do\n"
                f"    for T in {tok}; do\n"
                "      torchrun --nproc_per_node=8 "
                f"{os.path.relpath(__file__, _REPO_ROOT)} \\\n"
                "        --models $M --tokens $T --impl both --rounds 5 \\\n"
                "        --rtol 0.06 --fused-rtol 0.06 --csv /tmp/c_${M}_${T}.csv\n"
                "    done\n"
                "  done\n\n"
                "or pass --no-graph for the eager-only sweep (every cell in "
                "one process), or --graph explicitly to override this check."
            )
        if captures_per_cell and cells > 1 and ctx.is_main:
            # Repeated capture in one process eventually kills a rank. The
            # whole-layer captures --graph does are fairly robust (13 cells x 2
            # measured fine); the per-leg ones are not (5 in one process
            # SIGSEGVs about half the time, on both collective backends, so it
            # is the capture path and not either backend). Say what the run is
            # about to do so a mid-sweep SIGSEGV is recognisable rather than
            # mysterious.
            print(
                f"[NOTE] capturing {captures_per_cell} CUDA graphs per cell "
                f"across {cells} cells. If a rank dies mid-sweep, that is the "
                "capture path giving out -- split the run by model, or one "
                "cell per process.",
                flush=True,
            )
        if ctx.is_main:
            print(
                f"[ENV] gfx={gfx} cu={get_cu_num()} tp={args.tp} "
                f"route={args.route} quant=a4w4 "
                f"baseline_comm={args.comm_backend.describe()} "
                f"fmoe_csv={AITER_CONFIGS.AITER_CONFIG_FMOE_FILE}",
                flush=True,
            )
            if not MegaMoeTP.is_available():
                print(
                    f"[INFO] fused MegaMoeTP unavailable: "
                    f"{MegaMoeTP.unavailable_reason()}",
                    flush=True,
                )

        bad = [t for t in args.tokens if t <= 0 or t % args.tp]
        if bad:
            raise ValueError(f"tokens {bad} must be positive multiples of TP={args.tp}")
        tokens = sorted(set(args.tokens))
        max_local = max(tokens) // args.tp

        rows: list[dict] = []
        for name in args.models:
            shape = MODELS[name]
            if shape.a4w4_untuned and ctx.is_main:
                print(
                    f"[WARN] {shape.name}: {shape.tuned_csv} is tuned for a8w4, so the "
                    "a4w4 sweep falls back to untuned kernel selection.",
                    flush=True,
                )
            if ctx.is_main:
                print(f"\n=== {shape.tag(args.tp)} ===", flush=True)
            weights = None
            try:
                weights = build_sharded_weights(shape, ctx, args.tp, args.seed)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
            if not all_ranks_ok(weights is not None, ctx.device):
                del weights
                torch.cuda.empty_cache()
                if ctx.is_main:
                    print(
                        f"[SKIP] {shape.name}: out of memory building the weight "
                        f"shard ({_free_gib(ctx.device):.1f} GiB free). This node is "
                        "shared; retry with fewer --models or "
                        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.",
                        flush=True,
                    )
                continue
            barrier()
            try:
                for global_tokens in tokens:
                    row = None
                    skip = ""
                    try:
                        row = run_case(
                            shape, weights, ctx, args, global_tokens, max_local
                        )
                    except SkipCase as exc:
                        # Already unanimous across ranks; no further vote.
                        skip = str(exc)
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                    if not skip and not all_ranks_ok(row is not None, ctx.device):
                        skip = "out of memory"
                    if skip:
                        if ctx.is_main:
                            print(
                                f"[SKIP] {shape.name} M={global_tokens}: {skip}",
                                flush=True,
                            )
                        torch.cuda.empty_cache()
                        continue
                    rows.append(row)
                    torch.cuda.empty_cache()
                    if ctx.is_main:
                        if args.no_perf:
                            print(
                                f"[TP-MOE] {shape.name} M={global_tokens} "
                                f"m={row['local_tokens']} "
                                f"rel_l2={row['rel_l2']:.6f}",
                                flush=True,
                            )
                        else:
                            # Lead with the captured numbers when they exist:
                            # they are the result, and the eager legs behind
                            # them are the breakdown of where launches went.
                            head = (
                                f"[TP-MOE] {shape.name} M={global_tokens} "
                                f"m={row['local_tokens']} "
                                f"rel_l2={row['rel_l2']:.6f} "
                            )
                            if "split_graph_us" in row:
                                head += (
                                    f"GRAPH split={row['split_graph_us']:.1f} "
                                    f"fused={row.get('fused_graph_us', float('nan')):.1f} "
                                    f"speedup={row.get('graph_speedup', float('nan')):.3f} | "
                                    f"eager split={row['e2e_us']:.1f} "
                                    f"fused={row.get('fused_us', float('nan')):.1f} "
                                    f"speedup={row.get('fused_speedup', float('nan')):.3f} | "
                                )
                            print(
                                head
                                + f"ag={row['ag_us']:.1f} g1={row['gemm1_us']:.1f} "
                                f"g2={row['gemm2_us']:.1f} "
                                f"other={row['moe_other_us']:.1f} "
                                f"rs={row['rs_us']:.1f} "
                                f"e2e={row['e2e_us']:.1f}us "
                                f"comm={row['comm_pct']:.0f}%"
                                + _suspect_note(row, global_tokens),
                                flush=True,
                            )
                    barrier()
            finally:
                del weights
                torch.cuda.empty_cache()

        if ctx.is_main and rows:
            df = pd.DataFrame(rows)
            cols = [c for c in _PERF_COLUMNS if c in df.columns]
            extra = [c for c in df.columns if c not in cols]
            print("\n" + "=" * 100)
            print("TP MoE (AllGather + GEMM1 + GEMM2 + ReduceScatter), a4w4")
            print("=" * 100)
            print(df[cols].to_markdown(index=False, floatfmt=".3f"))
            if args.csv:
                df[cols + extra].to_csv(args.csv, index=False)
                print(f"\nwrote {args.csv}")
            _report_non_monotonic(df)
            print("\nkernels selected:")
            print(
                df[["model", "global_tokens", "gemm1_kernel", "gemm2_kernel"]]
                .drop_duplicates(subset=["model", "gemm1_kernel", "gemm2_kernel"])
                .to_markdown(index=False)
            )
        barrier()
        if ctx.is_main:
            print(f"\nTP_MOE_UT_OK cases={len(rows)}", flush=True)
        return 0
    finally:
        cleanup_dist()


if __name__ == "__main__":
    sys.exit(main())
