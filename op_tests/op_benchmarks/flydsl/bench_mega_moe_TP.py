# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP MoE (MXFP4, gfx950): split baseline vs fused MegaMoE, including communication.

Reproduce the M3 TP2 / M=2048 case from ROCm/aiter#5937::

    HIP_VISIBLE_DEVICES=0,1 HSA_ENABLE_IPC_MODE_LEGACY=1 \
        torchrun --standalone --nproc_per_node=2 \
        op_tests/op_benchmarks/flydsl/bench_mega_moe_TP.py \
        --e2e --models m3 --tokens 2048 --tp 2 --csv /tmp/m3_tp2.csv

For M3 TP4 balanced routed experts with the shared expert always on, use
``run_m3_tp4_balanced_e2e.sh``. It keeps the full router work in the timed path.

Adapted from the benchmark deleted in 9aeac9a602. M is the GLOBAL token count;
with ag_rs each rank has M/TP input/output rows. Experts are replicated and the
intermediate dimension is sharded. Both implementations use CUDA Graph replay;
the reported time is the fastest round's slowest rank. Compilation, input/weight
initialization, accuracy checks and graph capture are outside the timed region.
Large-batch launch parameters follow AITER_MEGAMOE_TP_LB_* (or --lb-* overrides).
The PR table measures --e2e: normalization, routing, MoE and the next-layer
normalization/FP8 output. Without --e2e only the TP MoE itself is measured.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import gc
import json
import logging
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# Preserve the deleted benchmark's environment for each measurement scope.
_ENV = {"AITER_USE_SYSTEM_TRITON": "1"}
if "--e2e" in sys.argv:
    _ENV.update(
        AITER_QUICK_REDUCE_QUANTIZATION="INT4",
        AITER_MEGAMOE_TP_LB_MIN="256",
        AITER_MEGAMOE_TP_LB_MT="5",
        AITER_MEGAMOE_TP_LB_NPP="2",
        AITER_MEGAMOE_TP_LB_Q="3",
    )
else:
    _ENV.update(
        AITER_SITUV2_A4W4="1",
        AITER_FLYDSL_STAGE2_FP8="1",
        AITER_BF16_FP8_MOE_BOUND="0",
    )
for _key, _value in _ENV.items():
    os.environ.setdefault(_key, _value)

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter import biased_grouped_topk, dtypes, topk_gating
from aiter.dist.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_fused_allreduce_rmsnorm,
    tensor_model_parallel_fused_allreduce_rmsnorm_quant,
    tensor_model_parallel_reduce_scatter,
)
from aiter.fused_moe import (
    fused_moe_2stages,
    fused_topk,
    get_2stage_cfgs,
    get_padded_M,
    moe_sorting,
    stage2_uses_route_reduce,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.flydsl.kernels.mega_moe_tp.sp_rs_norm import SpRsNorm
from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP as MegaMoeTPLayer
from aiter.ops.flydsl.mega_moe_tp import MegaMoeTPConfig
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
    GateMode,
)
from aiter.ops.flydsl.mxfp4_kname import (
    MXFP4_G1_VARIANTS,
    _is_mxfp4_kname,
    _parse_mxfp4_g1_kname,
    parse_g2_kname_any,
)
from aiter.ops.quant import get_hip_quant
from aiter.ops.shuffle import shuffle_weight
from aiter.tuned_gemm import tgemm
from aiter.utility import fp4_utils

logger = logging.getLogger("aiter")
SUPPORTED_GFX = ("gfx950",)
REPLICATED = ("ar",)
QUANT_TYPE = aiter.QuantType.per_1x32
AQ_DTYPE = dtypes.fp4x2
WQ_DTYPE = dtypes.fp4x2


@dataclass(frozen=True)
class ModelShape:
    name: str
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    act_type: aiter.ActivationType
    x_scale: float = 1.0

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
    "kimi3": ModelShape("kimi3", 3584, 3072, 896, 16, aiter.ActivationType.Situv2),
    "dsv3": ModelShape("dsv3", 7168, 2048, 256, 8, aiter.ActivationType.Silu),
    "dsv4": ModelShape("dsv4", 7168, 3072, 384, 6, aiter.ActivationType.Silu),
    "glm5": ModelShape("glm5", 6144, 2048, 257, 9, aiter.ActivationType.Silu),
    "m3": ModelShape(
        "m3", 6144, 3072, 129, 5, aiter.ActivationType.Swiglu, x_scale=0.25
    ),
}


def _situ(shape: ModelShape):
    if shape.act_type == aiter.ActivationType.Situv2:
        return DEFAULT_SITUV2_BETA, DEFAULT_SITUV2_LINEAR_BETA
    return None


@dataclass
class DistCtx:
    rank: int
    world: int
    device: torch.device
    custom_comm_error: str = ""

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_dist() -> DistCtx:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    err = ""
    try:
        from aiter.ops.communication import init_dist_env

        if rank == 0:
            print("[TP-MOE] bringing up aiter parallel state", flush=True)
        init_dist_env(world, rank, local_rank=local_rank)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
        err = err or "aiter parallel state did not initialize"
    torch.set_default_device(device)
    return DistCtx(rank=rank, world=world, device=device, custom_comm_error=err)


def cleanup_dist() -> None:
    try:
        from aiter.ops.communication import destroy_dist_env

        destroy_dist_env()
    except Exception as exc:  # noqa: BLE001
        logger.debug("[TP-MOE] destroy_dist_env: %s", exc)
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    torch.cuda.synchronize()
    dist.barrier()


def _all_reduce_scalar(value: float, device, op) -> float:
    t = torch.tensor(float(value), dtype=torch.float32, device=device)
    dist.all_reduce(t, op=op)
    return float(t.item())


def all_ranks_ok(ok: bool, device) -> bool:
    return _all_reduce_scalar(int(ok), device, dist.ReduceOp.MIN) > 0


def time_us(fn, *, iters: int, warmup: int, device, rounds: int = 3) -> float:
    for _ in range(warmup):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    best = float("inf")
    for _ in range(max(1, rounds)):
        barrier()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        local = start.elapsed_time(end) / iters * 1000.0
        rmax = _all_reduce_scalar(local, device, dist.ReduceOp.MAX)
        best = min(best, rmax)
    return best


@dataclass
class TpMoeWeights:
    shape: ModelShape
    tp_size: int
    local_inter_dim: int
    w1: torch.Tensor
    w1_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    w1_ref: torch.Tensor
    w1_scale_ref: torch.Tensor
    w2_ref: torch.Tensor
    w2_scale_ref: torch.Tensor


def _quantize_experts_chunked(
    experts, rows, cols, magnitude, seed, device, chunk_bytes=512 << 20
):
    chunk = max(1, min(experts, chunk_bytes // max(rows * cols * 2, 1)))
    quant = aiter.get_torch_quant(QUANT_TYPE)
    qt = torch.empty((experts, rows, cols // 2), dtype=torch.uint8, device=device)
    scale = None
    gen = torch.Generator(device=device)
    for start in range(0, experts, chunk):
        n = min(chunk, experts - start)
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
    return qt.view(WQ_DTYPE), scale


def build_sharded_weights(
    shape: ModelShape, ctx: DistCtx, tp_size: int, seed: int
) -> TpMoeWeights:
    inter_r = shape.local_inter_dim(tp_size)
    H, E = shape.model_dim, shape.experts
    base = seed + 1_000_000 * ctx.rank
    w1_qt, w1_scale = _quantize_experts_chunked(
        E, 2 * inter_r, H, H**-0.25, base, ctx.device
    )
    w2_qt, w2_scale = _quantize_experts_chunked(
        E, H, inter_r, inter_r**-0.25, base + 7, ctx.device
    )
    torch.cuda.empty_cache()
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


@dataclass
class TpMoeInputs:
    x_local: torch.Tensor
    topk_weights_local: torch.Tensor
    topk_ids_local: torch.Tensor
    route_meta_local: torch.Tensor
    global_tokens: int
    local_tokens: int
    comm_mode: str = "ag_rs"


def _balanced_scores(
    rows: int, experts: int, topk: int, start: int, device
) -> torch.Tensor:
    score = torch.zeros((rows, experts), dtype=dtypes.bf16, device=device)
    for t in range(rows):
        end = start + topk
        if end <= experts:
            score[t, start:end] = 1.0
        else:
            score[t, start:] = 1.0
            score[t, : end - experts] = 1.0
        start = end % experts
    return score


def make_inputs(
    shape, ctx, tp_size, global_tokens, seed, route="balanced", comm_mode="ag_rs"
) -> TpMoeInputs:
    if global_tokens % tp_size:
        raise ValueError(
            f"global tokens={global_tokens} must be divisible by TP={tp_size}"
        )
    m = global_tokens // tp_size
    ar = comm_mode in REPLICATED
    E, K = shape.experts, shape.topk
    if ar:
        rows = global_tokens
        gen = torch.Generator(device=ctx.device).manual_seed(seed + 7919)
        x = torch.randn(
            (rows, shape.model_dim),
            dtype=torch.float32,
            device=ctx.device,
            generator=gen,
        )
        x = (x * shape.x_scale).to(dtypes.bf16)
        rgen = torch.Generator(device=ctx.device).manual_seed(seed + 104729)
        start = 0
    else:
        rows = m
        gen = torch.Generator(device=ctx.device).manual_seed(seed + ctx.rank)
        x = shape.x_scale * torch.randn(
            (rows, shape.model_dim), dtype=dtypes.bf16, device=ctx.device, generator=gen
        )
        rgen = gen
        start = (ctx.rank * m * K) % E
    if route == "balanced":
        score = _balanced_scores(rows, E, K, start, ctx.device)
    else:
        score = torch.randn(
            (rows, E), dtype=dtypes.bf16, device=ctx.device, generator=rgen
        )
    topk_weights, topk_ids = fused_topk(x, score, K, True)
    topk_weights, topk_ids = topk_weights.contiguous(), topk_ids.contiguous()
    meta = torch.empty((rows, 2 * K), dtype=torch.int32, device=ctx.device)
    meta[:, :K].copy_(topk_ids)
    meta[:, K:].copy_(topk_weights.view(torch.int32))
    return TpMoeInputs(
        x.contiguous(), topk_weights, topk_ids, meta, global_tokens, m, comm_mode
    )


class TpCollectives:
    def __init__(self, ctx: DistCtx, tp_size: int):
        self.tp_size = tp_size
        self.error = ctx.custom_comm_error
        self.enabled = not self.error
        self.fell_back = False

    @staticmethod
    def _as_float(x):
        if x.dtype not in (torch.uint8, torch.int8):
            return x
        if x.dim() != 2 or x.shape[-1] % 2:
            return None
        return x.view(torch.bfloat16)

    def all_gather(self, x, out):
        if (
            self.enabled
            and x.is_contiguous()
            and (x.numel() * x.element_size()) % 16 == 0
        ):
            view = self._as_float(x)
            if view is not None:
                with contextlib.suppress(Exception):
                    got = tensor_model_parallel_all_gather(view, use_custom=True, dim=0)
                    return got.view(x.dtype) if view.dtype != x.dtype else got
        self.fell_back |= self.enabled
        dist.all_gather_into_tensor(out, x)
        return out

    def reduce_scatter(self, x, out):
        ok = (
            self.enabled
            and x.is_contiguous()
            and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and x.shape[0] % self.tp_size == 0
            and x.numel() % (self.tp_size * (16 // x.element_size())) == 0
        )
        if ok:
            with contextlib.suppress(Exception):
                return tensor_model_parallel_reduce_scatter(x, use_custom=True, dim=0)
        self.fell_back |= self.enabled
        dist.reduce_scatter_tensor(out, x)
        return out

    def all_reduce(self, x, out):
        x = x.contiguous()
        if self.enabled and x.dtype in (torch.bfloat16, torch.float16, torch.float32):
            try:
                return tensor_model_parallel_all_reduce(x, prefill_support=True)
            except Exception:  # noqa: BLE001
                self.fell_back = True
        out.copy_(x)
        dist.all_reduce(out)
        return out

    def describe(self) -> str:
        if not self.enabled:
            return f"rccl ({self.error})"
        return "custom+rccl-fallback" if self.fell_back else "custom"


class TokenGather:
    def __init__(
        self, shape: ModelShape, tp_size: int, max_local_tokens: int, device, comm
    ):
        total = max_local_tokens * tp_size
        self.topk = shape.topk
        self.comm = comm
        self._x = torch.empty(
            (total, shape.model_dim), dtype=dtypes.bf16, device=device
        )
        self._meta = torch.empty(
            (total, 2 * shape.topk), dtype=torch.int32, device=device
        )
        self._w = torch.empty((total, shape.topk), dtype=torch.float32, device=device)
        self._i = torch.empty((total, shape.topk), dtype=torch.int32, device=device)

    def route(self, inputs: TpMoeInputs):
        g, k = inputs.global_tokens, self.topk
        meta = self.comm.all_gather(inputs.route_meta_local, out=self._meta[:g])
        w, i = self._w[:g], self._i[:g]
        i.copy_(meta[:, :k])
        w.view(torch.int32).copy_(meta[:, k:])
        return w, i

    def __call__(self, inputs: TpMoeInputs):
        x = self.comm.all_gather(inputs.x_local, out=self._x[: inputs.global_tokens])
        return (x, *self.route(inputs))


def _partial_keyword(fn, *keys: str) -> str:
    for _ in range(8):
        if fn is None:
            break
        kwargs = getattr(fn, "keywords", None) or {}
        for key in keys:
            if kwargs.get(key):
                return str(kwargs[key])
        fn = getattr(fn, "func", None)
    return ""


@dataclass(frozen=True)
class _TunedPlan:
    metadata: object
    block_m: int
    accumulate: bool
    prequant: bool
    gemm1_kernel: str
    gemm2_kernel: str

    @property
    def ag_wire(self) -> str:
        return "fp4_1x32" if self.prequant else "bf16"


def _gemm1_takes_prequantized_fp4(kname: str) -> bool:
    if not kname:
        return False
    if kname.startswith("flydsl_moe1_afp4_wfp4_"):
        return True
    if not _is_mxfp4_kname(kname):
        return False
    try:
        parsed = _parse_mxfp4_g1_kname(kname)
    except ValueError:
        return False
    if parsed["a_dtype"] != "fp4" or parsed["inline_quant"]:
        return False
    return (parsed["BM"], parsed["use_nt"], False) in MXFP4_G1_VARIANTS["fp4"]


class TunedPlans:
    def __init__(self, weights: TpMoeWeights):
        self.weights = weights
        self._cache: dict[int, _TunedPlan] = {}

    def __call__(self, global_tokens: int) -> _TunedPlan:
        plan = self._cache.get(global_tokens)
        if plan is None:
            shape = self.weights.shape
            md = get_2stage_cfgs(
                get_padded_M(global_tokens),
                shape.model_dim,
                self.weights.local_inter_dim,
                shape.experts,
                shape.topk,
                dtypes.bf16,
                AQ_DTYPE,
                WQ_DTYPE,
                QUANT_TYPE,
                True,
                shape.act_type,
                False,
                0,
                0,
                True,
                GateMode.SEPARATED.value,
            )
            k1 = _partial_keyword(md.stage1, "kernelName1", "kernelName")
            k2 = _partial_keyword(md.stage2, "kernelName2", "kernelName")
            if md.output_aux:
                accumulate = bool(parse_g2_kname_any(k2)["atomic"])
            else:
                accumulate = not stage2_uses_route_reduce(md.stage2)
            plan = _TunedPlan(
                md,
                int(md.block_m),
                accumulate,
                _gemm1_takes_prequantized_fp4(k1),
                k1,
                k2,
            )
            self._cache[global_tokens] = plan
        return plan


def sort_routes(shape: ModelShape, w_all, i_all, plan: _TunedPlan):
    md = plan.metadata
    args = (i_all, w_all, shape.experts, shape.model_dim, dtypes.bf16, plan.block_m)
    if md.output_aux:
        return moe_sorting(
            *args, accumulate=plan.accumulate, output_aux=md.output_aux, output=None
        )
    return (
        *moe_sorting(*args, accumulate=plan.accumulate, flat=md.flat, output=None),
        None,
        None,
    )


class SplitLocalGemms:
    def __init__(self, weights: TpMoeWeights):
        shape = weights.shape
        self.weights = weights
        situ = _situ(shape)
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
            "beta": situ[0] if situ else None,
            "linear_beta": situ[1] if situ else None,
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
            rev,
        ) = sorted_ret
        forced = (
            replace(plan.metadata, prequant=True) if plan.prequant else plan.metadata
        )
        return fused_moe_2stages(
            a1,
            self.weights.w1,
            self.weights.w2,
            self.weights.shape.topk,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            True,
            plan.block_m,
            a1_scale=a1_scale,
            topk_ids=i_all,
            topk_weights=w_all,
            m_indices=m_indices,
            reverse_sorted=rev,
            _metadata_transform=lambda _md: forced,
            **self.kwargs,
        )


class SplitTpMoe:
    def __init__(
        self, weights: TpMoeWeights, ctx: DistCtx, max_local_tokens: int, comm
    ):
        shape, tp = weights.shape, weights.tp_size
        self.shape = shape
        self.comm = comm
        self.plans = TunedPlans(weights)
        self.gather = TokenGather(shape, tp, max_local_tokens, ctx.device, comm)
        self.quant = get_hip_quant(QUANT_TYPE)
        self.gemms = SplitLocalGemms(weights)
        total = max_local_tokens * tp
        self._payload = torch.empty(
            (total, shape.model_dim // 2), dtype=torch.uint8, device=ctx.device
        )
        self._scale = torch.empty(
            (total, shape.model_dim // 32), dtype=torch.uint8, device=ctx.device
        )
        self._y = torch.empty(
            (max_local_tokens, shape.model_dim), dtype=dtypes.bf16, device=ctx.device
        )

    def kernel_names(self, global_tokens: int) -> tuple[str, str]:
        plan = self.plans(global_tokens)
        return plan.gemm1_kernel, plan.gemm2_kernel

    def __call__(self, inputs: TpMoeInputs) -> torch.Tensor:
        m, g = inputs.local_tokens, inputs.global_tokens
        plan = self.plans(g)
        w_all, i_all = self.gather.route(inputs)
        sorted_ret = sort_routes(self.shape, w_all, i_all, plan)
        if plan.prequant:
            xq, xq_scale = self.quant(inputs.x_local, quant_dtype=AQ_DTYPE)
            a1 = self.comm.all_gather(
                xq.view(torch.uint8).view(m, -1), out=self._payload[:g]
            ).view(AQ_DTYPE)
            a1_scale = self.comm.all_gather(
                xq_scale.view(torch.uint8).view(m, -1), out=self._scale[:g]
            ).view(dtypes.fp8_e8m0)
        else:
            a1, a1_scale = self.gather(inputs)[0], None
        partial = self.gemms(a1, a1_scale, w_all, i_all, sorted_ret, plan).contiguous()
        return self.comm.reduce_scatter(partial, out=self._y[:m])

    def local_moe(self, x, wts, ids, plan: _TunedPlan) -> torch.Tensor:
        sorted_ret = sort_routes(self.shape, wts, ids, plan)
        a1, a1_scale = (
            self.quant(x, quant_dtype=AQ_DTYPE) if plan.prequant else (x, None)
        )
        return self.gemms(a1, a1_scale, wts, ids, sorted_ret, plan)

    def warmup(self, x, wts, ids) -> None:
        self.local_moe(x, wts, ids, self.plans(int(x.shape[0])))


class SplitTpMoeAR(SplitTpMoe):
    def __init__(self, weights, ctx, max_local_tokens, comm):
        super().__init__(weights, ctx, max_local_tokens, comm)
        total = max_local_tokens * weights.tp_size
        H = weights.shape.model_dim
        self._yall = torch.empty((total, H), dtype=dtypes.bf16, device=ctx.device)

    def __call__(self, inputs: TpMoeInputs) -> torch.Tensor:
        g = inputs.global_tokens
        x = inputs.x_local
        partial = self.local_moe(
            x, inputs.topk_weights_local, inputs.topk_ids_local, self.plans(g)
        )
        return self.comm.all_reduce(partial, out=self._yall[: inputs.global_tokens])


class MegaMoeTP:
    def __init__(
        self,
        weights: TpMoeWeights,
        ctx: DistCtx,
        max_local_tokens: int,
        group=None,
        comm_mode="ag_rs",
        comm_dtype="fp8",
        ar_gather="bf16",
        schedule="dynamic",
    ):
        shape = weights.shape
        situ = _situ(shape)
        config = MegaMoeTPConfig(
            rank=ctx.rank,
            world_size=weights.tp_size,
            model_dim=shape.model_dim,
            inter_dim=weights.local_inter_dim,
            experts=shape.experts,
            topk=shape.topk,
            max_local_tokens=max_local_tokens,
            activation=shape.act_type,
            beta=situ[0] if situ else None,
            linear_beta=situ[1] if situ else None,
            comm_mode=comm_mode,
            comm_dtype=comm_dtype,
            ar_gather=ar_gather,
            schedule=schedule,
        )
        self.engine = MegaMoeTPLayer(
            config,
            w1=weights.w1,
            w1_scale=weights.w1_scale,
            w2=weights.w2,
            w2_scale=weights.w2_scale,
            group=group,
            device=ctx.device,
        )

    def __call__(self, inputs: TpMoeInputs) -> torch.Tensor:
        return self.engine(
            inputs.x_local, inputs.topk_weights_local, inputs.topk_ids_local
        )


@torch.no_grad()
def torch_partial(
    weights: TpMoeWeights, x_all, w_all, i_all, global_tokens: int
) -> torch.Tensor:
    shape = weights.shape
    situ = _situ(shape) or (1.0, 1.0)
    torch_quant = aiter.get_torch_quant(QUANT_TYPE)
    a1_qt, a1_scale = torch_quant(x_all, quant_dtype=AQ_DTYPE)
    out1 = torch_moe_stage1(
        a1_qt,
        weights.w1_ref,
        weights.w2_ref,
        w_all,
        i_all,
        dtype=dtypes.bf16,
        activation=shape.act_type,
        quant_type=QUANT_TYPE,
        a1_scale=a1_scale,
        w1_scale=weights.w1_scale_ref,
        w1_bias=None,
        doweight=False,
        swiglu_limit=None,
        situ_beta=situ[0],
        situ_linear_beta=situ[1],
    )
    a2_qt, a2_scale = torch_quant(out1, quant_dtype=AQ_DTYPE)
    return torch_moe_stage2(
        a2_qt.view(global_tokens, shape.topk, -1),
        weights.w1_ref,
        weights.w2_ref,
        w_all,
        i_all,
        dtype=dtypes.bf16,
        quant_type=QUANT_TYPE,
        w2_scale=weights.w2_scale_ref,
        a2_scale=a2_scale,
        w2_bias=None,
        doweight=True,
    )


@torch.no_grad()
def torch_reference(
    weights, ctx, inputs: TpMoeInputs, gather: TokenGather
) -> torch.Tensor:
    if inputs.comm_mode in REPLICATED:
        xs = inputs.x_local.float()
        x_all, w_all, i_all = (
            xs.to(dtypes.bf16),
            inputs.topk_weights_local,
            inputs.topk_ids_local,
        )
    else:
        x_all, w_all, i_all = (t.clone() for t in gather(inputs))
    acc = torch_partial(weights, x_all, w_all, i_all, inputs.global_tokens).float()
    dist.all_reduce(acc)
    if inputs.comm_mode == "ar":
        return acc
    m = inputs.local_tokens
    return acc[ctx.rank * m : (ctx.rank + 1) * m]


def rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    err = torch.sum((actual.float() - expected.float()) ** 2)
    ref = torch.sum(expected.float() ** 2)
    dist.all_reduce(err)
    dist.all_reduce(ref)
    return (
        float("nan")
        if float(ref.item()) == 0.0
        else float(torch.sqrt(err / ref).item())
    )


class SkipCase(Exception):
    pass


def new_row(
    shape, args, global_tokens, local_tokens, inter_local, knames, comm
) -> dict:
    return {
        "model": shape.name,
        "tp": args.tp,
        "comm_mode": args.comm_mode,
        "global_tokens": global_tokens,
        "local_tokens": local_tokens,
        "model_dim": shape.model_dim,
        "inter_dim_local": inter_local,
        "experts": shape.experts,
        "topk": shape.topk,
        "act": str(shape.act_type).split(".")[-1],
        "gemm1_kernel": knames[0],
        "gemm2_kernel": knames[1],
        "comm": comm,
        "rel_l2": float("nan"),
        "seed": args.seed,
        "iters": args.iters,
        "rounds": args.rounds,
        "capture_warmup": args.capture_warmup,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
    }


def gate(row: dict, key: str, value: float, tol: float, what: str) -> None:
    row[key] = value
    if not value < tol:
        raise AssertionError(
            f"{row['model']} M={row['global_tokens']}: {what} rel_l2={value:.6f} exceeds {tol}"
        )


def set_speedup(row: dict) -> None:
    f = row.get("fused_graph_us", float("nan"))
    row["graph_speedup"] = (
        row.get("split_graph_us", float("nan")) / f if f > 0 else float("nan")
    )


def print_row(row: dict, no_perf: bool) -> None:
    nan = float("nan")
    msg = (
        f"[TP-MOE] {row['model']} M={row['global_tokens']} m={row['local_tokens']} "
        f"rel_l2={row['rel_l2']:.6f} fused_rel_l2={row.get('fused_rel_l2', nan):.6f} "
        f"fused_ref_rel_l2={row.get('fused_ref_rel_l2', nan):.6f}"
    )
    if not no_perf:
        msg += (
            f" | split={row.get('split_graph_us', nan):.1f} us mega={row.get('fused_graph_us', nan):.1f} us "
            f"speedup={row.get('graph_speedup', nan):.3f}"
        )
    print(msg, flush=True)


_PERF_COLUMNS = [
    "model",
    "comm_mode",
    "global_tokens",
    "local_tokens",
    "inter_dim_local",
    "comm",
    "rel_l2",
    "ag_wire",
    "split_graph_us",
    "fused_graph_us",
    "graph_speedup",
    "split_graph_rel_l2",
    "fused_graph_rel_l2",
    "fused_rel_l2",
    "fused_ref_rel_l2",
]


def write_table(rows: list[dict], args, title: str) -> None:
    if rows:
        df = pd.DataFrame(rows)
        cols = [c for c in _PERF_COLUMNS if c in df.columns]
        print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)
        print(df[cols].to_markdown(index=False, floatfmt=".3f"))
        if args.csv:
            df[cols + [c for c in df.columns if c not in cols]].to_csv(
                args.csv, index=False
            )
            print(f"\nwrote {args.csv}")
    print(f"\nTP_MOE_UT_OK cases={len(rows)}", flush=True)


def _warmup_inputs(shape: ModelShape, global_tokens: int, device):
    x = torch.randn((global_tokens, shape.model_dim), dtype=dtypes.bf16, device=device)
    ids = (
        torch.arange(global_tokens * shape.topk, dtype=torch.int32, device=device)
        % shape.experts
    )
    wts = torch.full(
        (global_tokens, shape.topk),
        1.0 / shape.topk,
        dtype=torch.float32,
        device=device,
    )
    return x, ids.view(global_tokens, shape.topk), wts


def jit_warmup(
    moe: SplitTpMoe, shape: ModelShape, ctx: DistCtx, global_tokens: int
) -> bool:
    ok = True
    if ctx.is_main:
        try:
            x, ids, wts = _warmup_inputs(shape, global_tokens, ctx.device)
            moe.warmup(x, wts, ids)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[jit-warmup] %s M=%d: %s", shape.name, global_tokens, exc)
            ok = False
        torch.cuda.empty_cache()
    return all_ranks_ok(ok, ctx.device)


def _graph_capture_ctx():
    try:
        from aiter.dist.parallel_state import get_tp_group, graph_capture

        get_tp_group()
    except Exception:  # noqa: BLE001
        return contextlib.nullcontext()
    return graph_capture()


def _time_graph(impl, inputs, args, ctx) -> tuple[float, torch.Tensor | None]:
    graph = out = None
    try:
        for _ in range(args.capture_warmup):
            impl(inputs)
        torch.cuda.synchronize()
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
        us = time_us(
            graph.replay,
            iters=args.iters,
            warmup=args.warmup,
            device=ctx.device,
            rounds=args.rounds,
        )
        graph.replay()
        torch.cuda.synchronize()
        if isinstance(out, tuple):
            return us, tuple(t.clone() for t in out)
        return us, out.clone()
    finally:
        graph = out = None
        gc.collect()
        torch.cuda.synchronize()


def run_case(shape, weights, ctx, args, global_tokens, max_local_tokens) -> dict:
    tp = args.tp
    inputs = make_inputs(
        shape, ctx, tp, global_tokens, args.seed, args.route, args.comm_mode
    )
    moe = (SplitTpMoeAR if args.comm_mode in REPLICATED else SplitTpMoe)(
        weights, ctx, max_local_tokens, args.comm_backend
    )
    if not jit_warmup(moe, shape, ctx, global_tokens):
        raise SkipCase(
            "kernel build failed on rank 0 (see the [jit-warmup] warning above)"
        )
    row = new_row(
        shape,
        args,
        global_tokens,
        inputs.local_tokens,
        weights.local_inter_dim,
        moe.kernel_names(global_tokens),
        "",
    )
    check_ref = global_tokens <= args.accuracy_max_tokens
    y = moe(inputs).clone()
    barrier()
    if check_ref:
        gate(
            row,
            "rel_l2",
            rel_l2(y, torch_reference(weights, ctx, inputs, moe.gather)),
            args.rtol,
            "split vs torch",
        )
        torch.cuda.empty_cache()
    if _all_reduce_scalar(
        int(bool(torch.isnan(y).any())), ctx.device, dist.ReduceOp.MAX
    ):
        raise AssertionError(f"{shape.tag(tp)} tokens={global_tokens}: output has NaN")

    fused = None
    if args.impl in ("fused", "both"):
        fused = MegaMoeTP(
            weights,
            ctx,
            max_local_tokens,
            comm_mode=args.comm_mode,
            comm_dtype=args.comm_dtype,
            ar_gather=args.ar_gather,
        )
        fused.engine.prepare([inputs.local_tokens])
        row["mega_launch_config"] = str(fused.engine.launch_config(inputs.local_tokens))
        row["mega_lb_q"] = fused.engine.engine.lb_q
        row["comm_dtype"] = args.comm_dtype
        if ctx.is_main:
            print(
                f"[MEGA] {row['mega_launch_config']} lb_q={row['mega_lb_q']}",
                flush=True,
            )
        gate(
            row,
            "fused_rel_l2",
            rel_l2(fused(inputs), y),
            args.fused_rtol,
            "fused vs split",
        )
        err = fused.engine.poll_errors()
        if not all_ranks_ok(err == 0, ctx.device):
            raise AssertionError(
                f"{shape.tag(tp)} tokens={global_tokens}: fused watchdog fired "
                f"(rank {ctx.rank}: {err})"
            )
        if check_ref:
            ref = torch_reference(weights, ctx, inputs, moe.gather)
            gate(
                row,
                "fused_ref_rel_l2",
                rel_l2(fused(inputs), ref),
                args.fused_ref_rtol,
                "fused vs torch",
            )
            del ref
            torch.cuda.empty_cache()
    if not args.no_perf:
        row["ag_wire"] = moe.plans(global_tokens).ag_wire
        row["split_graph_us"], rep = _time_graph(moe, inputs, args, ctx)
        gate(row, "split_graph_rel_l2", rel_l2(rep, y), args.rtol, "split graph replay")
        if fused is not None:
            row["fused_graph_us"], rep = _time_graph(fused, inputs, args, ctx)
            gate(
                row,
                "fused_graph_rel_l2",
                rel_l2(rep, fused(inputs)),
                args.rtol,
                "fused graph replay",
            )
            if not all_ranks_ok(fused.engine.poll_errors() == 0, ctx.device):
                raise AssertionError("MegaMoE watchdog fired during graph timing")
            set_speedup(row)
    row["comm"] = args.comm_backend.describe()
    return row


@dataclass(frozen=True)
class E2eModel:
    experts: int
    topk: int
    scale: float
    shared_w: float
    eps: float
    gemma: bool
    logits_fp32: bool
    swiglu_limit: float | None = None
    bias_glm: bool = False


E2E_MODELS = {
    "m3": E2eModel(
        128, 4, 2.0, 1.0, 1e-6, gemma=True, logits_fp32=False, swiglu_limit=7.0
    ),
    "glm5": E2eModel(
        256, 8, 2.5, 1.0, 1e-5, gemma=False, logits_fp32=True, bias_glm=True
    ),
}


@dataclass
class E2eInputs:
    part: torch.Tensor
    residual: torch.Tensor
    w_post: torch.Tensor
    w_next: torch.Tensor
    wg: torch.Tensor
    bias: torch.Tensor
    global_tokens: int
    local_tokens: int


E2E_RESIDUAL_SCALE = 16.0


def make_e2e_inputs(shape, ctx, tp, global_tokens, seed) -> E2eInputs:
    cfg = E2E_MODELS[shape.name]
    E = cfg.experts
    H = shape.model_dim
    gen = torch.Generator(device=ctx.device).manual_seed(seed + 31 * ctx.rank)
    shared = torch.Generator(device=ctx.device).manual_seed(seed + 7)
    part = (
        torch.randn((global_tokens, H), generator=gen, device=ctx.device)
        * (E2E_RESIDUAL_SCALE * tp**-0.5)
    ).to(dtypes.bf16)
    residual = (
        E2E_RESIDUAL_SCALE
        * torch.randn((global_tokens, H), generator=shared, device=ctx.device)
    ).to(dtypes.bf16)

    def norm_weight():
        noise = 0.1 * torch.randn((H,), generator=shared, device=ctx.device)
        return (shape.x_scale * (1.0 + noise) - (1.0 if cfg.gemma else 0.0)).to(
            dtypes.bf16
        )

    w_post = norm_weight()
    w_next = norm_weight()
    wg = (
        torch.randn((E, H), generator=shared, device=ctx.device) * (3 * H**-0.5)
    ).to(dtypes.bf16)
    if cfg.bias_glm:
        bias = 7.0 + 0.01 * torch.randn((E,), generator=shared, device=ctx.device)
    else:
        bias = 0.05 * torch.randn((E,), generator=shared, device=ctx.device)
    return E2eInputs(
        part,
        residual,
        w_post,
        w_next,
        wg,
        bias.float(),
        global_tokens,
        global_tokens // tp,
    )


class SplitE2e:
    def __init__(
        self,
        weights: TpMoeWeights,
        ctx: DistCtx,
        max_local_tokens: int,
        comm,
        bf16=False,
    ):
        self.bf16 = bf16
        self.cfg = cfg = E2E_MODELS[weights.shape.name]
        self.E, self.K, self.scale, self.eps = cfg.experts, cfg.topk, cfg.scale, cfg.eps
        self.moe = SplitTpMoeAR(weights, ctx, max_local_tokens, comm)
        self.moe.gemms.kwargs["swiglu_limit"] = cfg.swiglu_limit
        total = max_local_tokens * weights.tp_size
        self.ids = torch.full(
            (total, self.K + 1), self.E, dtype=torch.int32, device=ctx.device
        )
        self.tw = torch.full(
            (total, self.K + 1), cfg.shared_w, dtype=torch.float32, device=ctx.device
        )

    def __call__(self, inp: E2eInputs):
        T, K = inp.global_tokens, self.K
        gemma = self.cfg.gemma
        normed, res = tensor_model_parallel_fused_allreduce_rmsnorm(
            inp.part, inp.residual, inp.w_post, self.eps, gemma_norm=gemma
        )
        ids, tw = self.ids[:T], self.tw[:T]
        tw_r, ids_r = (
            torch.split(tw, [K, 1], dim=1)[0],
            torch.split(ids, [K, 1], dim=1)[0],
        )
        if self.cfg.logits_fp32:
            logits = tgemm.mm(normed, inp.wg, None, otype=dtypes.fp32)
            biased_grouped_topk(logits, inp.bias, tw_r, ids_r, 1, 1, True, self.scale)
        else:
            logits = tgemm.mm(normed, inp.wg, None, otype=dtypes.bf16)
            topk_gating(
                tw_r, ids_r, logits, inp.bias, True, self.scale, score_func="sigmoid"
            )
        out = tensor_model_parallel_fused_allreduce_rmsnorm_quant(
            self._moe(normed, T).contiguous(),
            res,
            inp.w_next,
            self.eps,
            quant_type="per_token",
            gemma_norm=gemma,
            emit_bf16=self.bf16,
        )
        q, res_out, s = out[:3]
        return (q, s, res_out) + tuple(out[3:])

    def _moe(self, normed, T):
        return self.moe.local_moe(normed, self.tw[:T], self.ids[:T], self.moe.plans(T))

    @torch.no_grad()
    def reference(self, inp: E2eInputs):
        acc = inp.part.float()
        dist.all_reduce(acc)
        res = (acc + inp.residual.float()).to(dtypes.bf16)
        normed = _norm(res, inp.w_post, self.eps, self.cfg.gemma)
        y = self._moe(normed, inp.global_tokens).float()
        dist.all_reduce(y)
        res_out = (y + res.float()).to(dtypes.bf16)
        return _norm(res_out, inp.w_next, self.eps, self.cfg.gemma).float(), res_out


def _norm(x, w, eps, gemma) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * ((1.0 + w.float()) if gemma else w.float())).to(dtypes.bf16)


class MegaE2e:
    def __init__(
        self,
        weights: TpMoeWeights,
        ctx: DistCtx,
        max_local_tokens: int,
        tokens,
        bf16=False,
    ):
        shape = weights.shape
        self.rank = ctx.rank
        self.bf16 = bf16
        cfg = E2E_MODELS[shape.name]
        self.K, self.eps = cfg.topk, cfg.eps
        self.fused = MegaMoeTP(
            weights,
            ctx,
            max_local_tokens,
            comm_mode="ag_rs",
            comm_dtype="fp8",
            schedule="dynamic",
        )
        self.fused.engine.engine.tn_eps = self.eps
        self.fused.engine.engine.tn_gemma = cfg.gemma
        self.fused.engine.prepare(
            sorted({t // weights.tp_size for t in tokens}),
            tail=not bf16,
            tail_bf16=bf16,
        )
        self.sprs = SpRsNorm(
            shape.model_dim,
            max_local_tokens * weights.tp_size,
            self.eps,
            device=ctx.device,
            router=(cfg.experts, cfg.topk, cfg.scale, cfg.shared_w),
            gemma=cfg.gemma,
            logit_bf16=not cfg.logits_fp32,
        )
        self.ids = torch.empty(
            (max_local_tokens, self.K + 1), dtype=torch.int32, device=ctx.device
        )
        self.tw = torch.empty(
            (max_local_tokens, self.K + 1), dtype=torch.float32, device=ctx.device
        )
        self.y = torch.empty(
            (max_local_tokens, shape.model_dim), dtype=dtypes.bf16, device=ctx.device
        )

    def __call__(self, inp: E2eInputs):
        m = inp.local_tokens
        rows = slice(self.rank * m, (self.rank + 1) * m)
        ids, tw = self.ids[:m], self.tw[:m]
        out, res = self.sprs(
            inp.part, inp.residual, inp.w_post, router=(inp.wg, inp.bias, ids, tw)
        )
        res_own = res[rows]
        _, q, s, *b = self.fused.engine.forward(
            out[rows],
            tw,
            ids,
            out=self.y[:m],
            tail=(res_own, res_own, inp.w_next),
            bf16=self.bf16,
        )
        return (q.view(dtypes.fp8), s, res_own) + tuple(b)


class MegaE2eSmall(MegaE2e):
    """Keep the fused LB tail when M/TP is too small for SpRsNorm routing."""

    def __init__(self, weights, ctx, max_local_tokens, tokens):
        super().__init__(weights, ctx, max_local_tokens, tokens)
        self.router_cfg = E2E_MODELS[weights.shape.name]
        self.ids[:, self.K].fill_(self.router_cfg.experts)
        self.tw[:, self.K].fill_(self.router_cfg.shared_w)

    def __call__(self, inp: E2eInputs):
        if self.sprs.routes(inp.global_tokens):
            return super().__call__(inp)
        m = inp.local_tokens
        own = slice(self.rank * m, (self.rank + 1) * m)
        ids, tw = self.ids[:m], self.tw[:m]
        x, res = self.sprs(inp.part, inp.residual, inp.w_post)
        logits = tgemm.mm(x[own], inp.wg, None, otype=dtypes.bf16)
        topk_gating(
            tw[:, :self.K], ids[:, :self.K], logits, inp.bias,
            True, self.router_cfg.scale, score_func="sigmoid",
        )
        res_own = res[own]
        _, q, s, *bf16 = self.fused.engine.forward(
            x[own], tw, ids, out=self.y[:m],
            tail=(res_own, res_own, inp.w_next), bf16=self.bf16,
        )
        return (q.view(dtypes.fp8), s, res_own) + tuple(bf16)


@contextlib.contextmanager
def balanced_routed_ids_only(split, mega, ids_global, ids_local):
    """Change only the IDs consumed by MoE; leave router work and weights timed."""
    original_local = split.moe.local_moe
    original_forward = mega.fused.engine.forward

    def local(x, weights, ids, plan):
        assert x.shape[0] == ids_global.shape[0]
        return original_local(x, weights, ids_global, plan)

    def forward(x, weights, ids, *args, **kwargs):
        assert x.shape[0] == ids_local.shape[0]
        return original_forward(x, weights, ids_local, *args, **kwargs)

    split.moe.local_moe = local
    mega.fused.engine.forward = forward
    try:
        yield
    finally:
        split.moe.local_moe = original_local
        mega.fused.engine.forward = original_forward


def balanced_m3_ids(global_tokens, ctx):
    ids = torch.empty((global_tokens, 5), dtype=torch.int32, device=ctx.device)
    ids[:, :4] = (
        torch.arange(global_tokens * 4, dtype=torch.int32, device=ctx.device)
        .view(global_tokens, 4) % 128
    )
    ids[:, 4] = 128
    local = global_tokens // ctx.world
    return ids, ids[ctx.rank * local:(ctx.rank + 1) * local].contiguous()


def _dequant(q, s) -> torch.Tensor:
    return q.float() * s.float().view(-1, 1)


@torch.no_grad()
def run_case_e2e_balanced_m3_tp4(
    shape, weights, ctx, args, global_tokens, max_local_tokens, tokens
) -> dict:
    """Full e2e with 128 balanced routed experts and one always-on shared expert."""
    if shape.name != "m3" or args.tp != 4:
        raise ValueError("balanced-routed-keep-shared requires M3 TP4")
    M, m = global_tokens, global_tokens // args.tp
    inp = make_e2e_inputs(shape, ctx, args.tp, M, args.seed)
    split = SplitE2e(weights, ctx, max_local_tokens, args.comm_backend)
    if not jit_warmup(split.moe, shape, ctx, M):
        raise SkipCase("Split JIT warmup failed")
    mega = (
        MegaE2e(weights, ctx, max_local_tokens, tokens) if M >= 64
        else MegaE2eSmall(weights, ctx, max_local_tokens, tokens)
    )
    cfg = mega.fused.engine.launch_config(m)
    if not cfg.lb or not mega.fused.engine.tail_ok(m):
        raise AssertionError(f"M={M}: expected LB and fused tail, got {cfg}")

    ids, local_ids = balanced_m3_ids(M, ctx)
    own = slice(ctx.rank * m, (ctx.rank + 1) * m)
    counts = torch.bincount(ids[:, :4].flatten().long(), minlength=128)
    if int(counts.max()) - int(counts.min()) > 1:
        raise AssertionError("routed expert fixture is not balanced")
    if ctx.is_main:
        print(
            f"[BALANCED] M={M} routed_min/max={int(counts.min())}/{int(counts.max())} "
            f"shared={M} mega={cfg} lb_q={mega.fused.engine.engine.lb_q}",
            flush=True,
        )

    with balanced_routed_ids_only(split, mega, ids, local_ids):
        q0, s0, r0 = (t.clone() for t in split(inp))
        q1, s1, r1 = (t.clone() for t in mega(inp))
        barrier()
        if not bool((split.tw[:M, -1] == 1.0).all()):
            raise AssertionError("Split shared expert weight changed")
        if not bool((mega.tw[:m, -1] == 1.0).all()):
            raise AssertionError("Mega shared expert weight changed")
        ref_q, ref_r = split.reference(inp)
        row = new_row(
            shape, args, M, m, weights.local_inter_dim,
            split.moe.kernel_names(M), args.comm_backend.describe(),
        )
        row.update(
            comm_mode="e2e", route_variant="balanced_routed_keep_shared",
            capacity_token=max(tokens), expert_ids_override=True,
            router_weights="computed_by_original_router", e2e_route_match=1.0,
            routed_load_min=int(counts.min()), routed_load_max=int(counts.max()),
            routed_active_experts=int((counts > 0).sum()),
            shared_expert_load=M, mega_launch_config=str(cfg),
            mega_effective_lb_min=mega.fused.engine.engine.lb_min,
            mega_lb_q=mega.fused.engine.engine.lb_q,
            mega_router_impl="sprs_fused" if mega.sprs.routes(M) else "local_gemm_topk",
            tail_impl="mega_fused",
        )
        row["e2e_split_q_rel_l2"] = rel_l2(_dequant(q0, s0), ref_q)
        row["e2e_split_res_rel_l2"] = rel_l2(r0[own].float(), ref_r[own].float())
        gate(row, "e2e_q_rel_l2", rel_l2(_dequant(q1, s1), ref_q),
             args.e2e_rtol, "Mega next-layer input")
        gate(row, "e2e_res_rel_l2", rel_l2(r1.float(), ref_r[own].float()),
             args.e2e_rtol, "Mega residual")
        if not all_ranks_ok(mega.fused.engine.poll_errors() == 0, ctx.device):
            raise AssertionError("MegaMoE watchdog fired during accuracy check")
        if not all_ranks_ok(mega.sprs.poll_errors() == 0, ctx.device):
            raise AssertionError("SpRsNorm watchdog fired during accuracy check")
        row["split_e2e_us"], _ = _time_graph(split, inp, args, ctx)
        row["mega_e2e_us"], _ = _time_graph(mega, inp, args, ctx)
        if not all_ranks_ok(mega.fused.engine.poll_errors() == 0, ctx.device):
            raise AssertionError("MegaMoE watchdog fired during graph timing")
        if not all_ranks_ok(mega.sprs.poll_errors() == 0, ctx.device):
            raise AssertionError("SpRsNorm watchdog fired during graph timing")
        row["e2e_speedup"] = row["split_e2e_us"] / row["mega_e2e_us"]
        row["watchdog"] = 0
    return row


def run_case_e2e(
    shape, weights, ctx, args, global_tokens, max_local_tokens, tokens
) -> dict:
    tp = args.tp
    if global_tokens % (16 * tp):
        raise SkipCase(f"the fused router needs tokens divisible by {16 * tp}")
    inp = make_e2e_inputs(shape, ctx, tp, global_tokens, args.seed)
    split = SplitE2e(
        weights, ctx, max_local_tokens, args.comm_backend, bf16=args.e2e_tail_bf16
    )
    if not jit_warmup(split.moe, shape, ctx, global_tokens):
        raise SkipCase(
            "kernel build failed on rank 0 (see the [jit-warmup] warning above)"
        )
    row = new_row(
        shape,
        args,
        global_tokens,
        inp.local_tokens,
        weights.local_inter_dim,
        split.moe.kernel_names(global_tokens),
        "",
    )
    row["comm_mode"] = "e2e"
    mega = MegaE2e(weights, ctx, max_local_tokens, tokens, bf16=args.e2e_tail_bf16)
    row["mega_launch_config"] = str(mega.fused.engine.launch_config(inp.local_tokens))
    row["mega_lb_q"] = mega.fused.engine.engine.lb_q
    if ctx.is_main:
        print(f"[MEGA] {row['mega_launch_config']} lb_q={row['mega_lb_q']}", flush=True)
    q0, s0, r0, *b0 = split(inp)
    q1, s1, r1, *b1 = mega(inp)
    barrier()
    m, K = inp.local_tokens, mega.K
    rows = slice(ctx.rank * m, (ctx.rank + 1) * m)
    same = (
        mega.ids[:m, :K].sort(dim=1).values == split.ids[rows][:, :K].sort(dim=1).values
    ).all(dim=1)
    same_all = torch.empty((global_tokens,), dtype=same.dtype, device=same.device)
    dist.all_gather_into_tensor(same_all, same)
    row["e2e_route_match"] = float(same_all.float().mean())
    if row["e2e_route_match"] < args.e2e_route_match:
        raise AssertionError(
            f"{shape.tag(tp)} tokens={global_tokens}: routers agree on "
            f"{row['e2e_route_match']:.4f} of the tokens (< {args.e2e_route_match})"
        )
    ref_q, ref_r = split.reference(inp)
    keep = same_all.view(-1, 1)
    own = same.view(-1, 1)
    if b1:
        row["e2e_bf16_rel_l2"] = rel_l2(b1[0].float() * keep, ref_q * keep)
        row["e2e_split_bf16_rel_l2"] = rel_l2(b0[0].float(), ref_q)
    row["e2e_split_q_rel_l2"] = rel_l2(_dequant(q0, s0), ref_q)
    row["e2e_split_res_rel_l2"] = rel_l2(r0[rows].float(), ref_r[rows].float())
    gate(
        row,
        "e2e_q_rel_l2",
        rel_l2(_dequant(q1, s1) * keep, ref_q * keep),
        args.e2e_rtol,
        "mega next-layer input vs reference",
    )
    gate(
        row,
        "e2e_res_rel_l2",
        rel_l2(r1.float() * own, ref_r[rows].float() * own),
        args.e2e_rtol,
        "mega residual vs reference",
    )
    err = mega.fused.engine.poll_errors()
    if not all_ranks_ok(err == 0, ctx.device):
        raise AssertionError(
            f"{shape.tag(tp)} tokens={global_tokens}: fused watchdog fired"
        )
    if not all_ranks_ok(mega.sprs.poll_errors() == 0, ctx.device):
        raise AssertionError("SpRsNorm watchdog fired during e2e accuracy check")
    if not args.no_perf:
        row["split_e2e_us"], _ = _time_graph(split, inp, args, ctx)
        row["mega_e2e_us"], _ = _time_graph(mega, inp, args, ctx)
        if not all_ranks_ok(mega.fused.engine.poll_errors() == 0, ctx.device):
            raise AssertionError("MegaMoE watchdog fired during e2e graph timing")
        if not all_ranks_ok(mega.sprs.poll_errors() == 0, ctx.device):
            raise AssertionError("SpRsNorm watchdog fired during e2e graph timing")
        row["e2e_speedup"] = row["split_e2e_us"] / row["mega_e2e_us"]
    row["comm"] = args.comm_backend.describe()
    return row


def main_e2e(args) -> int:
    ctx = setup_dist()
    args.tp = args.tp or ctx.world
    args.comm_backend = TpCollectives(ctx, args.tp)
    try:
        if ctx.custom_comm_error:
            raise RuntimeError(
                f"--e2e needs aiter's parallel state: {ctx.custom_comm_error}"
            )
        if args.tp != ctx.world:
            raise ValueError(f"--tp={args.tp} must equal WORLD_SIZE={ctx.world}")
        tokens, max_local = _check_tokens(args)
        models = [m for m in args.models if m in E2E_MODELS]
        if ctx.is_main:
            print(
                f"[ENV] e2e gfx={get_gfx()} tp={args.tp} models={models} "
                f"(ATOM decoder-layer MoE region, CUDA graph)",
                flush=True,
            )
        rows: list[dict] = []
        for name in models:
            shape = MODELS[name]
            weights = build_sharded_weights(shape, ctx, args.tp, args.seed)
            barrier()
            for global_tokens in tokens:
                row, skip = None, ""
                try:
                    runner = (
                        run_case_e2e_balanced_m3_tp4
                        if args.route == "balanced-routed-keep-shared"
                        else run_case_e2e
                    )
                    row = runner(
                        shape, weights, ctx, args, global_tokens, max_local, tokens
                    )
                except SkipCase as exc:
                    skip = str(exc)
                torch.cuda.empty_cache()
                if skip:
                    if ctx.is_main:
                        print(f"[SKIP] {name} M={global_tokens}: {skip}", flush=True)
                    continue
                rows.append(row)
                if ctx.is_main:
                    print(
                        f"[E2E] {name} M={global_tokens} split={row.get('split_e2e_us', float('nan')):.1f} "
                        f"mega={row.get('mega_e2e_us', float('nan')):.1f} "
                        f"speedup={row.get('e2e_speedup', float('nan')):.3f} "
                        f"route_match={row['e2e_route_match']:.4f} "
                        f"vs ref: split q/res {row['e2e_split_q_rel_l2']:.4f}/{row['e2e_split_res_rel_l2']:.4f} "
                        f"mega q/res {row['e2e_q_rel_l2']:.4f}/{row['e2e_res_rel_l2']:.4f}"
                        + (
                            f" bf16 split/mega {row['e2e_split_bf16_rel_l2']:.4f}/{row['e2e_bf16_rel_l2']:.4f}"
                            if "e2e_bf16_rel_l2" in row
                            else ""
                        ),
                        flush=True,
                    )
                barrier()
            del weights
            torch.cuda.empty_cache()
        barrier()
        if ctx.is_main and rows:
            df = pd.DataFrame(rows)
            cols = [
                "model",
                "global_tokens",
                "local_tokens",
                "split_e2e_us",
                "mega_e2e_us",
                "e2e_speedup",
                "e2e_route_match",
                "e2e_split_q_rel_l2",
                "e2e_q_rel_l2",
                "e2e_split_res_rel_l2",
                "e2e_res_rel_l2",
                "e2e_split_bf16_rel_l2",
                "e2e_bf16_rel_l2",
                "comm",
            ]
            cols = [c for c in cols if c in df.columns]
            print(
                "\n"
                + "=" * 100
                + "\nATOM decoder-layer MoE region: split vs mega (us)\n"
                + "=" * 100
            )
            print(df[cols].to_markdown(index=False, floatfmt=".3f"))
            if args.csv:
                df.to_csv(args.csv, index=False)
                print(f"\nwrote {args.csv}")
        if ctx.is_main:
            print(f"\nTP_MOE_E2E_OK cases={len(rows)}", flush=True)
        return 0 if rows else 1
    finally:
        cleanup_dist()


def main_torchrun(args) -> int:
    ctx = setup_dist()
    args.tp = args.tp or ctx.world
    args.comm_backend = TpCollectives(ctx, args.tp)
    try:
        if args.tp != ctx.world:
            raise ValueError(f"--tp={args.tp} must equal WORLD_SIZE={ctx.world}")
        gfx = get_gfx()
        if gfx not in SUPPORTED_GFX:
            if ctx.is_main:
                print(f"[SKIP] a4w4 MoE needs {SUPPORTED_GFX}, found {gfx}")
            return 0
        tokens, max_local = _check_tokens(args)
        if ctx.is_main:
            print(
                f"[ENV] gfx={gfx} cu={get_cu_num()} tp={args.tp} quant=a4w4 comm_mode={args.comm_mode} "
                f"baseline_comm={args.comm_backend.describe()} fmoe_csv={AITER_CONFIGS.AITER_CONFIG_FMOE_FILE}",
                flush=True,
            )
        rows: list[dict] = []
        for name in args.models:
            shape = MODELS[name]
            if ctx.is_main:
                print(f"\n=== {shape.tag(args.tp)} ===", flush=True)
            weights = None
            try:
                weights = build_sharded_weights(shape, ctx, args.tp, args.seed)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
            if not all_ranks_ok(weights is not None, ctx.device):
                if ctx.is_main:
                    print(
                        f"[SKIP] {shape.name}: out of memory building the weight shard",
                        flush=True,
                    )
                continue
            barrier()
            for global_tokens in tokens:
                row, skip = None, ""
                try:
                    row = run_case(shape, weights, ctx, args, global_tokens, max_local)
                except SkipCase as exc:
                    skip = str(exc)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                if not skip and not all_ranks_ok(row is not None, ctx.device):
                    skip = "out of memory"
                torch.cuda.empty_cache()
                if skip:
                    if ctx.is_main:
                        print(
                            f"[SKIP] {shape.name} M={global_tokens}: {skip}", flush=True
                        )
                    continue
                rows.append(row)
                if ctx.is_main:
                    print_row(row, args.no_perf)
                barrier()
            del weights
            torch.cuda.empty_cache()
        barrier()
        if ctx.is_main:
            write_table(rows, args, "TP MoE, a4w4")
        return 0 if rows else 1
    finally:
        cleanup_dist()


def _check_tokens(args):
    bad = [t for t in args.tokens if t <= 0 or t % args.tp]
    if bad:
        raise ValueError(f"tokens {bad} must be positive multiples of TP={args.tp}")
    tokens = sorted(set(args.tokens))
    return tokens, max(tokens) // args.tp


_STAGE_NCTA = 256
_STAGE_SLOTS = 6
_STAGE_RING = 64
_STAGE_TICKS_PER_US = 100.0  # gfx950 s_memrealtime: 100 MHz


def _stage_capture(obj, inp, warmup):
    for _ in range(warmup):
        obj(inp)
    torch.cuda.synchronize()
    barrier()
    graph = torch.cuda.CUDAGraph()
    with _graph_capture_ctx(), torch.cuda.graph(graph):
        obj(inp)
    torch.cuda.synchronize()
    return graph


def _stage_case(args, ctx, with_split):
    if args.models != ["m3"] or args.tp != 4 or len(args.tokens) != 1:
        raise ValueError("stage timing requires one M3 TP4 token count")
    M = args.tokens[0]
    m = M // 4
    args.comm_backend = TpCollectives(ctx, args.tp)
    shape = MODELS["m3"]
    weights = build_sharded_weights(shape, ctx, args.tp, args.seed)
    inp = make_e2e_inputs(shape, ctx, 4, M, args.seed)
    split = SplitE2e(weights, ctx, m, args.comm_backend) if with_split else None
    if split is not None and not jit_warmup(split.moe, shape, ctx, M):
        raise RuntimeError("Split JIT warmup failed")
    mega = MegaE2e(weights, ctx, m, [M]) if M >= 64 else MegaE2eSmall(weights, ctx, m, [M])
    cfg = mega.fused.engine.launch_config(m)
    if not cfg.lb or not mega.fused.engine.tail_ok(m):
        raise AssertionError(f"M={M}: expected LB and fused tail, got {cfg}")
    ids, local_ids = balanced_m3_ids(M, ctx)
    return M, m, inp, split, mega, ids, local_ids


@torch.no_grad()
def stage_trace_main(args):
    from torch.profiler import ProfilerActivity, profile

    ctx = setup_dist()
    try:
        if ctx.world != 4 or ctx.custom_comm_error:
            raise RuntimeError(f"stage trace requires TP4: {ctx.custom_comm_error}")
        M, _, inp, split, mega, ids, local_ids = _stage_case(args, ctx, True)
        out = args.stage_output_dir
        out.mkdir(parents=True, exist_ok=True)
        with balanced_routed_ids_only(split, mega, ids, local_ids):
            for mode, obj in (("split", split), ("mega", mega)):
                graph = _stage_capture(obj, inp, args.capture_warmup)
                for _ in range(10):
                    graph.replay()
                torch.cuda.synchronize()
                barrier()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    for _ in range(5):
                        graph.replay()
                    torch.cuda.synchronize()
                path = out / f"{mode}_rank{ctx.rank}_graph_trace.json"
                prof.export_chrome_trace(str(path))
                barrier()
                del graph
                if ctx.is_main:
                    print(f"[TRACE] M={M} {mode}: {path}", flush=True)
        if not all_ranks_ok(mega.fused.engine.poll_errors() == 0, ctx.device):
            raise AssertionError("MegaMoE watchdog fired during trace")
        if not all_ranks_ok(mega.sprs.poll_errors() == 0, ctx.device):
            raise AssertionError("SpRsNorm watchdog fired during trace")
        return 0
    finally:
        cleanup_dist()


def _install_stage_stamps():
    """Instrument the Mega builder in this process, before constructing MegaE2e."""
    import flydsl.expr as fx
    from flydsl.expr import gpu
    from flydsl.expr.typing import T as fx_types
    from aiter.ops.flydsl.kernels.mega_moe_tp import mega_moe_tp_kernel as mk
    from aiter.ops.flydsl.kernels.mega_moe_tp.common import i32, now, traced
    from aiter.ops.flydsl.kernels.mega_moe_tp.mega_moe_tp_config import CTRL_EPB, CTRL_INTS

    base = CTRL_INTS

    @traced
    def stamp(a, slot, first, epoch):
        bid = i32(gpu.block_id("x"))
        idx = i32(base) + (
            ((epoch & i32(_STAGE_RING - 1)) * i32(_STAGE_NCTA) + bid)
            * i32(_STAGE_SLOTS) + i32(slot)
        ) * i32(2)
        ptr = fx.inttoptr(
            fx.PointerType.get(fx_types.i64, fx.AddressSpace.Global, 8),
            fx.Int64(a["ctrl"]) + fx.Int64(idx) * fx.Int64(4),
        )
        if first:
            cur = fx.Int64(fx.ptr_load(ptr, result_type=fx_types.i64))
            if cur == fx.Int64(0):
                fx.ptr_store(now().ir_value(), ptr)
        else:
            fx.ptr_store(now().ir_value(), ptr)

    build_comm = mk.build_communication
    build_gemm = mk.build_gemm

    def comm(kc):
        funcs = build_comm(kc)
        entry, exit_ = funcs["init_lds"], funcs["finish"]

        @traced
        def init_lds(L, tid, a):
            if tid == i32(0):
                bid = i32(gpu.block_id("x"))
                ptr = fx.inttoptr(
                    fx.PointerType.get(fx_types.i32, fx.AddressSpace.Global, 4),
                    fx.Int64(a["ctrl"]) + fx.Int64((i32(CTRL_EPB) + bid) * i32(4)),
                )
                epoch = i32(fx.ptr_load(ptr, result_type=fx_types.i32)) + i32(1)
                stamp(a, 0, True, epoch)
            entry(L, tid, a)

        @traced
        def finish(tid, a, epoch):
            exit_(tid, a, epoch)
            if tid == i32(0):
                stamp(a, 5, False, epoch)

        funcs["init_lds"], funcs["finish"] = init_lds, finish
        return funcs

    def gemm(kc):
        funcs = build_gemm(kc)
        g1, g2 = funcs["gemm1"], funcs["gemm2"]

        @traced
        def gemm1(L, tid, a, expert, i0, nnb, rows=None):
            if tid == i32(0):
                stamp(a, 1, True, a["epoch"])
            g1(L, tid, a, expert, i0, nnb, rows)
            if tid == i32(0):
                stamp(a, 2, False, a["epoch"])

        @traced
        def gemm2(L, tid, a, expert, ks0, r0, rows, signal, NKS, pidx, gi_lo=None, *gargs, **gkw):
            if tid == i32(0):
                stamp(a, 3, True, a["epoch"])
            g2(L, tid, a, expert, ks0, r0, rows, signal, NKS, pidx, gi_lo, *gargs, **gkw)
            if tid == i32(0):
                stamp(a, 4, False, a["epoch"])

        funcs["gemm1"], funcs["gemm2"] = gemm1, gemm2
        return funcs

    mk.build_communication = comm
    mk.build_gemm = gemm
    return base


def _stage_from_stamps(raw):
    entries = raw.tolist()
    vals = [[row[i] for row in entries if row[i] > 0] for i in range(_STAGE_SLOTS)]
    if not all(vals):
        raise RuntimeError(f"missing Mega timestamps: {[len(v) for v in vals]}")
    t = [min(vals[0]), min(vals[1]), max(vals[2]), min(vals[3]), max(vals[4]), max(vals[5])]
    return dict(
        kernel_stamp_us=(t[5] - t[0]) / _STAGE_TICKS_PER_US,
        kernel_to_first_g1_us=(t[1] - t[0]) / _STAGE_TICKS_PER_US,
        first_g1_to_last_g2_us=(t[4] - t[1]) / _STAGE_TICKS_PER_US,
        last_g2_to_kernel_exit_us=(t[5] - t[4]) / _STAGE_TICKS_PER_US,
        g1_span_us=(t[2] - t[1]) / _STAGE_TICKS_PER_US,
        g2_span_us=(t[4] - t[3]) / _STAGE_TICKS_PER_US,
        overlap_g1_g2_us=max(0, t[2] - t[3]) / _STAGE_TICKS_PER_US,
    )


@contextlib.contextmanager
def _balanced_mega_only(mega, local_ids):
    original = mega.fused.engine.forward

    def forward(x, weights, ids, *args, **kwargs):
        assert x.shape[0] == local_ids.shape[0]
        return original(x, weights, local_ids, *args, **kwargs)

    mega.fused.engine.forward = forward
    try:
        yield
    finally:
        mega.fused.engine.forward = original


@torch.no_grad()
def stage_stamp_main(args):
    base = _install_stage_stamps()
    ctx = setup_dist()
    try:
        if ctx.world != 4 or ctx.custom_comm_error:
            raise RuntimeError(f"stage stamps require TP4: {ctx.custom_comm_error}")
        M, m, inp, _, mega, _, local_ids = _stage_case(args, ctx, False)
        eng = mega.fused.engine.engine
        eng.ctrl = torch.zeros(
            base + _STAGE_RING * _STAGE_NCTA * _STAGE_SLOTS * 2,
            dtype=torch.int32, device=ctx.device,
        )
        eng._args = (None, None)
        if eng.ctrl.data_ptr() % 8:
            raise AssertionError("timestamp buffer is not 8-byte aligned")
        with _balanced_mega_only(mega, local_ids):
            graph = _stage_capture(mega, inp, 80)
        if eng._args[1][0][10] != eng.ctrl.data_ptr():
            raise AssertionError("instrumented launcher did not use timestamp buffer")
        stamp_rows = []
        graph_avg_us = []
        for round_idx in range(3):
            barrier()
            eng.ctrl[base:].zero_()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            graph_avg_us.append(start.elapsed_time(end) * 1000 / 20)
            raw = eng.ctrl[base:].view(torch.int64).view(
                _STAGE_RING, _STAGE_NCTA, _STAGE_SLOTS
            ).cpu()
            seen = 0
            for epoch_slot, block in enumerate(raw):
                if not bool((block != 0).any()):
                    continue
                row = _stage_from_stamps(block)
                row.update(rank=ctx.rank, round=round_idx, epoch_slot=epoch_slot)
                stamp_rows.append(row)
                seen += 1
            if seen != 20:
                raise AssertionError((ctx.rank, round_idx, seen))
        gathered = [None] * ctx.world
        dist.all_gather_object(gathered, dict(rows=stamp_rows, graph_avg_us=graph_avg_us))
        if not all_ranks_ok(eng.poll_errors() == 0, ctx.device):
            raise AssertionError("MegaMoE watchdog fired during stamps")
        if not all_ranks_ok(mega.sprs.poll_errors() == 0, ctx.device):
            raise AssertionError("SpRsNorm watchdog fired during stamps")
        if ctx.is_main:
            args.stage_output_dir.mkdir(parents=True, exist_ok=True)
            path = args.stage_output_dir / "mega_stamps.json"
            path.write_text(json.dumps(dict(
                source_commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
                ).strip(),
                global_tokens=M, lb_min=M, tp=4,
                mega_launch_config=str(mega.fused.engine.launch_config(m)),
                mega_lb_q=eng.lb_q,
                gpu_visibility=os.environ.get("HIP_VISIBLE_DEVICES"),
                tick_per_us=_STAGE_TICKS_PER_US,
                ranks=gathered,
            ), indent=2))
            print(f"[STAMPS] M={M} ranks={len(gathered)} {path}", flush=True)
        return 0
    finally:
        cleanup_dist()


def _stage_trace_rows(path, mode):
    events = [
        event for event in json.loads(path.read_text())["traceEvents"]
        if event.get("cat") == "kernel" and event.get("ph") == "X"
    ]
    if not events or len(events) % 5:
        raise AssertionError((path, len(events)))
    per_graph = len(events) // 5
    chunks = [events[i:i + per_graph] for i in range(0, len(events), per_graph)]
    names = [[event["name"] for event in chunk] for chunk in chunks]
    if not all(row == names[0] for row in names[1:]):
        raise AssertionError(f"graph kernel sequence changed: {path}")
    rows = []
    mega_name = None
    for chunk in chunks[1:]:  # skip the profiler's first replay
        start = min(event["ts"] for event in chunk)
        end = max(event["ts"] + event["dur"] for event in chunk)
        row = {"trace_e2e_us": end - start}
        if mode == "split":
            g1 = [event for event in chunk if event["name"].startswith(
                ("mfma_moe1_", "gemm1_", "flydsl_moe1_", "flydsl_mxmoe_g1_")
            )]
            g2 = [event for event in chunk if event["name"].startswith(
                ("mfma_moe2_", "gemm2_", "flydsl_moe2_")
            )]
            if len(g1) != 1 or len(g2) != 1:
                raise AssertionError((path, [event["name"] for event in chunk]))
            g1_start = g1[0]["ts"]
            g2_end = g2[0]["ts"] + g2[0]["dur"]
            row.update(pre_g1_us=g1_start - start,
                       g1_to_g2_end_us=g2_end - g1_start,
                       post_g2_us=end - g2_end)
        else:
            mega = [event for event in chunk
                    if event["name"].startswith("mega_moe_tp_fused_")]
            if len(mega) != 1:
                raise AssertionError((path, [event["name"] for event in chunk]))
            mega_name = mega[0]["name"]
            row.update(before_mega_kernel_us=mega[0]["ts"] - start,
                       after_mega_kernel_us=end - (mega[0]["ts"] + mega[0]["dur"]))
        rows.append(row)
    return ({key: float(statistics.median(r[key] for r in rows)) for key in rows[0]},
            mega_name)


def stage_analyze_main(args):
    if args.models != ["m3"] or args.tp != 4 or len(args.tokens) != 1:
        raise ValueError("stage analysis requires one M3 TP4 token count")
    M = args.tokens[0]
    out = args.stage_output_dir
    with Path(args.csv).open(newline="") as file:
        paired = list(csv.DictReader(file))
    if len(paired) != 1 or paired[0]["route_variant"] != "balanced_routed_keep_shared":
        raise AssertionError("expected one balanced M3 TP4 e2e row")
    row = paired[0]
    stamps = json.loads((out / "mega_stamps.json").read_text())
    if (int(row["global_tokens"]) != M or stamps["global_tokens"] != M
            or stamps["tp"] != 4 or int(row["mega_lb_q"]) != stamps["mega_lb_q"]
            or row["mega_launch_config"] != stamps["mega_launch_config"]):
        raise AssertionError("e2e/stamp launch configurations differ")
    split_rows, mega_rows = [], []
    for rank in range(4):
        split_trace, _ = _stage_trace_rows(out / f"split_rank{rank}_graph_trace.json", "split")
        mega_trace, name = _stage_trace_rows(out / f"mega_rank{rank}_graph_trace.json", "mega")
        q_match = re.search(r"_q(\d+)(?:_|$)", name)
        trace_q = int(q_match.group(1)) if q_match else 1
        mt_match = re.search(r"_mt(\d+)_", name)
        if (mt_match is None or trace_q != stamps["mega_lb_q"]
                or f"mt={int(mt_match.group(1))}" not in stamps["mega_launch_config"]):
            raise AssertionError(f"trace/stamp launch configurations differ: {name}")
        split_rows.append(split_trace)
        mega_rows.append(mega_trace)
    split_rank = max(range(4), key=lambda rank: split_rows[rank]["trace_e2e_us"])
    mega_rank = max(range(4), key=lambda rank: mega_rows[rank]["trace_e2e_us"])
    split = split_rows[split_rank]
    trace = mega_rows[mega_rank]
    stamp_rows = stamps["ranks"][mega_rank]["rows"]
    if len(stamp_rows) != 60:
        raise AssertionError(f"expected 60 stamp samples, got {len(stamp_rows)}")
    stamp = {key: float(statistics.median(item[key] for item in stamp_rows))
             for key in ("kernel_to_first_g1_us", "first_g1_to_last_g2_us",
                         "last_g2_to_kernel_exit_us")}
    mega_pre = trace["before_mega_kernel_us"] + stamp["kernel_to_first_g1_us"]
    mega_middle = stamp["first_g1_to_last_g2_us"]
    mega_post = stamp["last_g2_to_kernel_exit_us"] + trace["after_mega_kernel_us"]
    summary = dict(
        global_tokens=M, local_tokens=M // 4,
        mega_launch_config=stamps["mega_launch_config"],
        mega_lb_q=stamps["mega_lb_q"],
        split_selected_rank=split_rank, mega_selected_rank=mega_rank,
        split_stages_us=dict(pre_g1=split["pre_g1_us"],
                             g1_to_g2_end=split["g1_to_g2_end_us"],
                             post_g2=split["post_g2_us"]),
        mega_stages_us=dict(pre_g1=mega_pre, g1_to_g2_end=mega_middle,
                            post_g2=mega_post),
        paired_split_e2e_us=float(row["split_e2e_us"]),
        paired_mega_e2e_us=float(row["mega_e2e_us"]),
        trace_split_e2e_us=split["trace_e2e_us"],
        trace_mega_e2e_us=trace["trace_e2e_us"],
    )
    path = out / "stage_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    mt = re.search(r"mt=(\d+)", stamps["mega_launch_config"]).group(1)
    print(
        f"[STAGE] M={M} mt={mt} "
        f"lb_q={stamps['mega_lb_q']} "
        f"Split(pre/middle/post)={split['pre_g1_us']:.1f}/"
        f"{split['g1_to_g2_end_us']:.1f}/{split['post_g2_us']:.1f} us "
        f"Mega(pre/middle/post)={mega_pre:.1f}/{mega_middle:.1f}/{mega_post:.1f} us "
        f"paired_e2e(S/M)={summary['paired_split_e2e_us']:.1f}/"
        f"{summary['paired_mega_e2e_us']:.1f} us",
        flush=True,
    )
    print("[STAGE] middle = first GEMM1 start to last GEMM2 end; includes scheduling and communication."
          " Profiler, CTA stamps, and paired e2e are separate runs.", flush=True)
    print(f"Stage summary: {path}", flush=True)
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--models", nargs="+", default=["m3"], choices=list(MODELS))
    p.add_argument(
        "--tokens", type=int, nargs="+", default=[2048], help="GLOBAL tokens"
    )
    p.add_argument("--tp", type=int, default=0, help="0 = torchrun WORLD_SIZE")
    p.add_argument("--comm-mode", choices=["ag_rs", "ar"], default="ag_rs")
    p.add_argument("--impl", choices=["unfused", "both"], default="both")
    p.add_argument("--comm-dtype", choices=["fp8", "bf16"], default="fp8")
    p.add_argument("--ar-gather", choices=["bf16", "fp8", "auto"], default="auto")
    p.add_argument(
        "--route", choices=["random", "balanced", "balanced-routed-keep-shared"],
        default="random",
    )
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--capture-warmup", type=int, default=400)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--accuracy-max-tokens",
        type=int,
        default=128,
        help="Run the slow torch reference up to this GLOBAL token count",
    )
    p.add_argument("--rtol", type=float, default=0.06)
    p.add_argument("--fused-rtol", type=float, default=0.08)
    p.add_argument("--fused-ref-rtol", type=float, default=0.0)
    p.add_argument("--no-perf", action="store_true", help="Accuracy only")
    p.add_argument("--csv", default=None)
    p.add_argument("--stage-mode", choices=("e2e", "trace", "stamp", "analyze"),
                   default="e2e")
    p.add_argument("--stage-output-dir", type=Path)
    p.add_argument(
        "--e2e",
        action="store_true",
        help="Full decoder-layer MoE region: AR+norm, router, MoE, AR+norm+FP8",
    )
    p.add_argument("--e2e-rtol", type=float, default=0.15)
    p.add_argument("--e2e-tail-bf16", action="store_true")
    p.add_argument("--e2e-route-match", type=float, default=0.9)
    p.add_argument("--lb-min", type=int, default=None)
    p.add_argument("--lb-mt", default=None, help="Rows/16, or Ti:Ni,...,N")
    p.add_argument("--lb-npp", type=int, default=None)
    p.add_argument("--lb-q", type=int, default=None)
    args = p.parse_args(argv)
    if args.stage_mode != "e2e" and (not args.e2e or args.stage_output_dir is None
                                       or args.route != "balanced-routed-keep-shared"):
        p.error("stage trace/stamp/analyze requires --e2e, a stage output dir, and balanced routed IDs")
    if args.route == "balanced-routed-keep-shared" and not args.e2e:
        p.error("balanced-routed-keep-shared requires --e2e")
    if args.e2e and any(m not in E2E_MODELS for m in args.models):
        p.error("--e2e supports m3 and glm5")
    if min(args.iters, args.rounds) < 1 or min(args.warmup, args.capture_warmup) < 0:
        p.error("iters/rounds must be positive and warmup counts nonnegative")
    for key in ("min", "mt", "npp", "q"):
        value = getattr(args, "lb_" + key)
        if value is not None:
            os.environ["AITER_MEGAMOE_TP_LB_" + key.upper()] = str(value)
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.stage_mode == "analyze":
        return stage_analyze_main(args)
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "1")) < 2:
        raise SystemExit(
            "Launch with torchrun --standalone --nproc_per_node=2 (or more)"
        )
    if not args.fused_ref_rtol:
        args.fused_ref_rtol = 0.045 if args.comm_dtype == "fp8" else 0.01
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.stage_mode == "trace":
        return stage_trace_main(args)
    if args.stage_mode == "stamp":
        return stage_stamp_main(args)
    return main_e2e(args) if args.e2e else main_torchrun(args)


if __name__ == "__main__":
    sys.exit(main())
