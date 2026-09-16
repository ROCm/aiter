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
* ``UnfusedTpMoe``         -- the comparison implementation, composed from four
  *separately timeable* modules: ``AllGatherTokens``, GEMM1, GEMM2 (both taken
  from the very kernels ``test_moe_2stage.py`` drives through ``fused_moe``),
  and ``ReduceScatterOutput``.
* ``MegaMoeTP``            -- interface placeholder for the future fused kernel
  that will collapse all four into one launch.  Not implemented yet; see
  ``mega_moe_plan.txt`` at the repo root for the design.

Only a4w4 (MXFP4 activation x MXFP4 weight, ``QuantType.per_1x32``) is wired up
for now.

Usage
-----
    # all four models, default token sweep
    PYTHONPATH=$PWD AITER_USE_SYSTEM_TRITON=1 AITER_SITUV2_A4W4=1 \
    AITER_FLYDSL_STAGE2_FP8=1 \
    torchrun --nproc_per_node=8 op_tests/multigpu_tests/test_mega_moe_TP.py

    # one model, accuracy only
    torchrun --nproc_per_node=8 op_tests/multigpu_tests/test_mega_moe_TP.py \
        --models kimi3 --tokens 8 32 128 --no-perf

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
from dataclasses import dataclass

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.fused_moe import (
    fused_moe,
    fused_topk,
    get_2stage_cfgs,
    get_padded_M,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
    GateMode,
)
from aiter.ops.quant import get_hip_quant
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils

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

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_dist() -> DistCtx:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    torch.set_default_device(device)
    return DistCtx(rank=rank, world=world, device=device)


def cleanup_dist() -> None:
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
    return TpMoeInputs(
        x_local=x.contiguous(),
        topk_weights_local=topk_weights.contiguous(),
        topk_ids_local=topk_ids.contiguous(),
        global_tokens=global_tokens,
        local_tokens=m,
    )


# ---------------------------------------------------------------------------
# Module 1 / 4: AllGather of the token shard
# ---------------------------------------------------------------------------
class AllGatherTokens:
    """AllGather the sequence-parallel activation shard into the full token set.

    Buffers are preallocated so the timed callable does no allocation, and so
    the same instance can be captured by a CUDA graph later.
    """

    def __init__(self, shape: ModelShape, tp_size: int, max_local_tokens: int, device):
        self.tp_size = tp_size
        self.model_dim = shape.model_dim
        self.topk = shape.topk
        total = max_local_tokens * tp_size
        self._x = torch.empty(
            (total, shape.model_dim), dtype=dtypes.bf16, device=device
        )
        self._w = torch.empty((total, shape.topk), dtype=torch.float32, device=device)
        self._i = torch.empty((total, shape.topk), dtype=torch.int32, device=device)
        # Routing metadata is tiny (topk*8 bytes/token) but must be gathered too:
        # every rank evaluates every global token against its own expert shard.
        # At M=8 a bare collective already costs ~40us, so issuing one per tensor
        # would make the baseline pay 3x the latency floor for no reason. Pack
        # ids and (bit-cast) weights into a single int32 payload instead.
        self._meta_local = torch.empty(
            (max_local_tokens, 2 * shape.topk), dtype=torch.int32, device=device
        )
        self._meta = torch.empty(
            (total, 2 * shape.topk), dtype=torch.int32, device=device
        )

    def wire_bytes(self, local_tokens: int) -> int:
        """Bytes this rank receives over the fabric (excluding its own shard)."""
        row = self.model_dim * 2 + self.topk * 8
        return local_tokens * row * (self.tp_size - 1)

    def __call__(self, inputs: TpMoeInputs):
        m, g, k = inputs.local_tokens, inputs.global_tokens, self.topk
        meta_local = self._meta_local[:m]
        meta_local[:, :k].copy_(inputs.topk_ids_local)
        meta_local[:, k:].copy_(inputs.topk_weights_local.view(torch.int32))

        dist.all_gather_into_tensor(self._x[:g], inputs.x_local)
        dist.all_gather_into_tensor(self._meta[:g], meta_local)

        x, w, i = self._x[:g], self._w[:g], self._i[:g]
        i.copy_(self._meta[:g, :k])
        w.view(torch.int32).copy_(self._meta[:g, k:])
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


class LocalMoeGemms:
    """The local two-stage MoE: sorting + activation quant + GEMM1 + GEMM2 + reduce.

    This calls ``fused_moe`` with exactly the kwargs ``test_moe_2stage.py`` uses,
    so the kernels selected here are the same ones that test benchmarks.  The
    individual GEMM1 / GEMM2 launch closures are captured through
    ``aiter.fused_moe.kernel_bench_callable`` (the hook behind that test's
    ``--kernel`` flag) so they can be timed in isolation.
    """

    def __init__(self, weights: TpMoeWeights):
        shape = weights.shape
        self.weights = weights
        self.shape = shape
        situ = shape.act_type == aiter.ActivationType.Situv2
        self.kwargs = {
            "w1_scale": weights.w1_scale,
            "w2_scale": weights.w2_scale,
            "quant_type": QUANT_TYPE,
            "activation": shape.act_type,
            "doweight_stage1": False,
            "intermediate_pad": 0,
            "hidden_pad": 0,
            "bias1": None,
            "bias2": None,
            # a4w4 leaves swiglu_limit unset (test_moe_2stage.py:970-973).
            "swiglu_limit": None,
            "beta": DEFAULT_SITUV2_BETA if situ else None,
            "linear_beta": DEFAULT_SITUV2_LINEAR_BETA if situ else None,
            # (fp4x2, fp4x2) resolves to SEPARATED (test_moe_2stage.py:953-968).
            "gate_mode": GateMode.SEPARATED.value,
        }

    def __call__(self, x_all, topk_weights, topk_ids):
        return fused_moe(
            x_all,
            self.weights.w1,
            self.weights.w2,
            topk_weights,
            topk_ids,
            **self.kwargs,
        )

    def capture_stage_callables(self, x_all, topk_weights, topk_ids):
        """Run one eager pass and return {"stage1": call, "stage2": call, "out": out}."""
        captured: list = []
        aiter.fused_moe.kernel_bench_callable = captured
        try:
            out = self(x_all, topk_weights, topk_ids)
        finally:
            aiter.fused_moe.kernel_bench_callable = None
        stages = dict(captured)
        return stages, out

    def kernel_names(self, global_tokens: int) -> tuple[str, str]:
        """Report which tuned GEMM1/GEMM2 kernels this shape resolves to."""
        shape = self.shape
        try:
            meta = get_2stage_cfgs(
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
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            return (f"<{type(exc).__name__}>", "")
        # MOEMetadata does not carry the names as fields: get_2stage_cfgs bakes
        # them into the functools.partial keywords of stage1/stage2
        # (fused_moe.py:2621-2627).
        return (
            _partial_keyword(meta.stage1, "kernelName1", "kernelName"),
            _partial_keyword(meta.stage2, "kernelName2", "kernelName"),
        )


# ---------------------------------------------------------------------------
# Module 4 / 4: ReduceScatter of the partial output
# ---------------------------------------------------------------------------
class ReduceScatterOutput:
    """Sum the per-rank partial outputs across TP and scatter back to the token shard."""

    def __init__(self, shape: ModelShape, tp_size: int, max_local_tokens: int, device):
        self.tp_size = tp_size
        self.model_dim = shape.model_dim
        self._y = torch.empty(
            (max_local_tokens, shape.model_dim), dtype=dtypes.bf16, device=device
        )

    def wire_bytes(self, local_tokens: int) -> int:
        return local_tokens * self.model_dim * 2 * (self.tp_size - 1)

    def __call__(self, partial: torch.Tensor, local_tokens: int) -> torch.Tensor:
        out = self._y[:local_tokens]
        dist.reduce_scatter_tensor(out, partial.contiguous())
        return out


# ---------------------------------------------------------------------------
# Composed baseline: AG -> GEMM1 -> GEMM2 -> RS
# ---------------------------------------------------------------------------
class UnfusedTpMoe:
    """The comparison implementation: four separate modules, four+ kernel launches."""

    name = "unfused"

    def __init__(self, weights: TpMoeWeights, ctx: DistCtx, max_local_tokens: int):
        self.weights = weights
        self.shape = weights.shape
        self.ctx = ctx
        self.tp_size = weights.tp_size
        self.allgather = AllGatherTokens(
            weights.shape, weights.tp_size, max_local_tokens, ctx.device
        )
        self.gemms = LocalMoeGemms(weights)
        self.reduce_scatter = ReduceScatterOutput(
            weights.shape, weights.tp_size, max_local_tokens, ctx.device
        )

    def __call__(self, inputs: TpMoeInputs) -> torch.Tensor:
        x_all, w_all, i_all = self.allgather(inputs)
        partial = self.gemms(x_all, w_all, i_all)
        return self.reduce_scatter(partial, inputs.local_tokens)


# ---------------------------------------------------------------------------
# Placeholder for the fused kernel
# ---------------------------------------------------------------------------
class MegaMoeTpNotImplemented(NotImplementedError):
    pass


class MegaMoeTP:
    """Interface placeholder for the fused AG + GEMM1 + GEMM2 + RS MoE kernel.

    Drop-in contract
    ----------------
    ``MegaMoeTP`` must be interchangeable with :class:`UnfusedTpMoe`::

        moe = MegaMoeTP(weights, ctx, max_local_tokens=...)
        y_local = moe(inputs)          # [m, model_dim] bf16

    Intended internal structure (see ``mega_moe_plan.txt`` for the full design):

    K1 = quantize local shard -> P2P AllGather push -> global route planning
         -> GEMM1 -> activation -> a2 quantization
       * quantize before the AllGather: the wire row drops from ``H*2`` to
         ``H/2 + H/32`` bytes (3.77x) and each rank quantizes only its own m
         rows instead of all M.
       * destination offsets are *static* (rank p writes at ``p*m``), so the
         whole histogram / count-exchange / dynamic-base machinery that
         ``mega_moe/dispatch.py`` needs for EP all-to-all collapses into a
         fixed-stride push.

    K2 = GEMM2 -> weighted top-k reduction -> ReduceScatter -> cross-TP accumulate
       * reuse ``mega_moe_stage2.p2p_scatter_epilog`` with the owner mapping
         changed from "token's source rank" to ``row // m``.
       * ``comm_fused_moe``'s ``direct`` collective is a good model for the
         receive side: pull all TP partials rather than pushing, which removes
         the destination-side atomic counters entirely.

    Both kernels should assign CTA roles by atomic ticket and gate every flag on
    a monotone epoch counter, so CUDA graph replay stays correct without any
    buffer clearing.
    """

    name = "mega_moe_tp"

    # Flip to True (and implement ``__call__``) once the kernel lands.
    IMPLEMENTED = False

    def __init__(
        self,
        weights: TpMoeWeights,
        ctx: DistCtx,
        max_local_tokens: int,
        *,
        ag_wire_quant: str = "fp4_1x32",
        rs_wire_quant: str = "fp8_1x32",
    ):
        self.weights = weights
        self.shape = weights.shape
        self.ctx = ctx
        self.tp_size = weights.tp_size
        self.max_local_tokens = max_local_tokens
        self.max_global_tokens = max_local_tokens * weights.tp_size
        self.ag_wire_quant = ag_wire_quant
        self.rs_wire_quant = rs_wire_quant
        # Symmetric-heap workspace, route metadata and epoch counters will hang
        # off here.  Everything must be allocated once, in __init__, so that
        # forward() is allocation-free and CUDA-graph capturable.
        self.workspace: dict[str, torch.Tensor] = {}

    @classmethod
    def is_available(cls, shape: ModelShape | None = None, tp_size: int = 8) -> bool:
        """Whether the fused kernel can serve this shape on this machine."""
        if not cls.IMPLEMENTED:
            return False
        return os.environ.get("AITER_MEGA_MOE_TP", "0") == "1"

    @classmethod
    def unavailable_reason(cls) -> str:
        if not cls.IMPLEMENTED:
            return "MegaMoeTP.IMPLEMENTED is False (fused kernel not written yet)"
        if os.environ.get("AITER_MEGA_MOE_TP", "0") != "1":
            return "AITER_MEGA_MOE_TP != 1"
        return ""

    def __call__(self, inputs: TpMoeInputs) -> torch.Tensor:
        raise MegaMoeTpNotImplemented(
            "MegaMoeTP is an interface placeholder: "
            f"{self.unavailable_reason()}. Implement K1/K2 as described in the "
            "class docstring and mega_moe_plan.txt, then set IMPLEMENTED = True."
        )

    # -- stubs the implementation is expected to fill in --------------------
    def preload(self) -> None:
        """AOT-compile every token-bucket variant without launching anything."""
        raise MegaMoeTpNotImplemented("MegaMoeTP.preload")


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
    moe: UnfusedTpMoe, shape: ModelShape, ctx: DistCtx, global_tokens: int
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
            moe.gemms(x, wts, ids)
            del x, ids, wts
        except Exception as exc:  # noqa: BLE001 - warmup must never be fatal
            logger.warning("[jit-warmup] %s M=%d: %s", shape.name, global_tokens, exc)
            ok = False
        torch.cuda.empty_cache()
    return all_ranks_ok(ok, ctx.device)


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
    moe = UnfusedTpMoe(weights, ctx, max_local_tokens)
    kname1, kname2 = moe.gemms.kernel_names(global_tokens)
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

    if args.no_perf:
        return row

    # ---------------- per-module perf ----------------
    x_all, w_all, i_all = moe.allgather(inputs)
    stages, _ = moe.gemms.capture_stage_callables(x_all, w_all, i_all)
    partial = moe.gemms(x_all, w_all, i_all)

    timer = functools.partial(
        time_us,
        iters=args.iters,
        warmup=args.warmup,
        device=ctx.device,
        rounds=args.rounds,
    )
    ag_mean, ag_max = timer(lambda: moe.allgather(inputs))
    moe_mean, _ = timer(lambda: moe.gemms(x_all, w_all, i_all))
    g1_mean = g2_mean = float("nan")
    if "stage1" in stages:
        g1_mean, _ = timer(stages["stage1"])
    if "stage2" in stages:
        g2_mean, _ = timer(stages["stage2"])
    rs_mean, rs_max = timer(lambda: moe.reduce_scatter(partial, inputs.local_tokens))
    e2e_mean, e2e_max = timer(lambda: moe(inputs))

    row.update(
        {
            "ag_us": ag_mean,
            "gemm1_us": g1_mean,
            "gemm2_us": g2_mean,
            "moe_us": moe_mean,
            "moe_other_us": moe_mean - (g1_mean + g2_mean),
            "rs_us": rs_mean,
            "e2e_us": e2e_mean,
            "e2e_max_us": e2e_max,
            # e2e is eager, so it also carries the host-side dispatch cost of
            # chaining the modules (tuned-config lookup, moe_sorting wrapper,
            # per-launch Python). At small M that is a real part of the gap the
            # fused kernel closes, so keep it visible rather than hiding it.
            "host_gap_us": e2e_mean - (ag_mean + moe_mean + rs_mean),
            "comm_pct": 100.0 * (ag_mean + rs_mean) / e2e_mean if e2e_mean else 0.0,
            "ag_GBps": moe.allgather.wire_bytes(inputs.local_tokens) / ag_max / 1e3,
            "rs_GBps": moe.reduce_scatter.wire_bytes(inputs.local_tokens)
            / rs_max
            / 1e3,
        }
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
    if args.impl in ("fused", "both"):
        fused = MegaMoeTP(weights, ctx, max_local_tokens)
        if not MegaMoeTP.is_available(shape, tp):
            row["fused_us"] = float("nan")
            row["fused_note"] = MegaMoeTP.unavailable_reason()
        else:
            y_fused = fused(inputs)
            row["fused_rel_l2"] = rel_l2(y_fused, y_actual)
            fused_mean, _ = timer(lambda: fused(inputs))
            row["fused_us"] = fused_mean
            row["fused_speedup"] = e2e_mean / fused_mean if fused_mean else float("nan")

    # Everything above is function-local, so it is released on return; the
    # caller calls empty_cache() between cases.  Do not `del` the tensors the
    # timing lambdas captured.
    return row


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
_PERF_COLUMNS = [
    "model",
    "global_tokens",
    "local_tokens",
    "inter_dim_local",
    "rel_l2",
    "ag_us",
    "gemm1_us",
    "gemm2_us",
    "moe_other_us",
    "rs_us",
    "host_gap_us",
    "e2e_us",
    "comm_pct",
    "ag_GBps",
    "rs_GBps",
    # only present with --probe-quant-wire / --impl fused|both
    "ag_fp4_us",
    "ag_speedup",
    "ag_bytes_ratio",
    "quant_local_us",
    "fused_us",
    "fused_speedup",
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
    p.add_argument("--csv", default=None, help="Write the perf table here (rank 0).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ctx = setup_dist()
    args.tp = args.tp or ctx.world
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
        if ctx.is_main:
            print(
                f"[ENV] gfx={gfx} cu={get_cu_num()} tp={args.tp} "
                f"route={args.route} quant=a4w4 "
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
                            print(
                                f"[TP-MOE] {shape.name} M={global_tokens} "
                                f"m={row['local_tokens']} "
                                f"rel_l2={row['rel_l2']:.6f} "
                                f"ag={row['ag_us']:.1f} g1={row['gemm1_us']:.1f} "
                                f"g2={row['gemm2_us']:.1f} "
                                f"other={row['moe_other_us']:.1f} "
                                f"rs={row['rs_us']:.1f} "
                                f"e2e={row['e2e_us']:.1f}us "
                                f"comm={row['comm_pct']:.0f}%",
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
