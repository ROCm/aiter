# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark: tensor-parallel MoE layer (a4w4, MXFP4), split baseline vs the
fused MegaMoE TP kernel. (Correctness: op_tests/multigpu_tests/test_mega_moe_TP.py.)

Every (model, tokens) cell checks accuracy and times both paths under CUDA
graph replay:

* split baseline, ``--comm-mode ag_rs``: route AllGather -> sort -> quant ->
  AllGather -> tuned GEMM1 / GEMM2 (``fused_moe_2stages``) -> ReduceScatter;
  ``ar``: sort -> quant -> GEMMs -> AllReduce (input replicated);
  ``ar_ar``: AllReduce of the input partials, then as ``ar``.
* fused: :class:`aiter.ops.flydsl.mega_moe_tp.MegaMoeTP` (one kernel).
* torch reference: ``torch_moe_stage1`` / ``torch_moe_stage2`` as in
  ``test_moe_2stage.py``.

Gates: split vs torch, fused vs split, fused vs torch rel L2 < --rtol, no NaN,
graph replay == eager. Weights: each rank's inter-dim shard, preshuffled.

Usage::

    torchrun --nproc_per_node=8 op_tests/op_benchmarks/flydsl/bench_mega_moe_TP.py \\
        --models glm5 --tokens 256
    python op_tests/op_benchmarks/flydsl/bench_mega_moe_TP.py --single-process \\
        --tp 4 --models m3 --tokens 64 --comm-mode ar

The single-process mode (one process drives every GPU through peer access,
``p2p_collectives.py``) also checks masked expert ids and empty batches.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import logging
import os
import sys
import threading
from dataclasses import dataclass, replace
from typing import ClassVar

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if os.path.isdir(os.path.join(_REPO_ROOT, "aiter")) and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# the split baseline runs the fused kernel's recipe: MXFP4 activations
# (SiTUv2 too) and FP8 stage-2 route rows at every token count
for _key, _value in {
    "AITER_USE_SYSTEM_TRITON": "1",
    "AITER_SITUV2_A4W4": "1",
    "AITER_FLYDSL_STAGE2_FP8": "1",
    "AITER_BF16_FP8_MOE_BOUND": "0",
}.items():
    os.environ.setdefault(_key, _value)

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.dist.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
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
from aiter.utility import fp4_utils

logger = logging.getLogger("aiter")

SUPPORTED_GFX = ("gfx950",)
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
    except Exception as exc:  # noqa: BLE001 - degrade to RCCL
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
    except Exception as exc:  # noqa: BLE001 - teardown must not mask a failure
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
        if rmax < best:
            best = rmax
            mean = (
                _all_reduce_scalar(local, device, dist.ReduceOp.SUM)
                / dist.get_world_size()
            )
    return mean


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
    ar = comm_mode in ("ar", "ar_ar")
    E, K = shape.experts, shape.topk
    if ar:
        rows = global_tokens
        partial = comm_mode == "ar_ar"
        gen = torch.Generator(device=ctx.device).manual_seed(
            seed + 7919 * ((ctx.rank + 1) if partial else 1)
        )
        x = torch.randn(
            (rows, shape.model_dim),
            dtype=torch.float32,
            device=ctx.device,
            generator=gen,
        )
        x = (x * (shape.x_scale * (tp_size**-0.5 if partial else 1.0))).to(dtypes.bf16)
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

    _BYTE_VIEW: ClassVar[dict] = {
        torch.uint8: torch.bfloat16,
        torch.int8: torch.bfloat16,
    }

    def __init__(self, ctx: DistCtx, tp_size: int):
        self.tp_size = tp_size
        self.error = ctx.custom_comm_error
        self.enabled = not self.error
        self.fell_back = False

    def _as_float(self, x):
        want = self._BYTE_VIEW.get(x.dtype)
        if want is None:
            return x
        step = torch.finfo(want).bits // 8 // x.element_size()
        if x.dim() != 2 or x.shape[-1] % step:
            return None
        return x.view(want)

    def all_gather(self, x, out):
        if (
            self.enabled
            and x.is_contiguous()
            and (x.numel() * x.element_size()) % 16 == 0
        ):
            view = self._as_float(x)
            if view is not None:
                with contextlib.suppress(Exception):  # shapes the kernel declines
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
            with contextlib.suppress(Exception):  # shapes the kernel declines
                return tensor_model_parallel_reduce_scatter(x, use_custom=True, dim=0)
        self.fell_back |= self.enabled
        dist.reduce_scatter_tensor(out, x)
        return out

    def all_reduce(self, x, out):
        x = x.contiguous()
        if self.enabled and x.dtype in (torch.bfloat16, torch.float16, torch.float32):
            try:
                return tensor_model_parallel_all_reduce(x, prefill_support=True)
            except Exception:  # noqa: BLE001 - shape the kernel declines
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

    def _payload_ag(self, xq, xq_scale, inputs: TpMoeInputs):
        m, g = inputs.local_tokens, inputs.global_tokens
        payload = xq.view(torch.uint8).view(m, -1)
        scale = xq_scale.view(torch.uint8).view(m, -1)
        a1 = self.comm.all_gather(payload, out=self._payload[:g])
        a1_scale = self.comm.all_gather(scale, out=self._scale[:g])
        return a1.view(AQ_DTYPE), a1_scale.view(dtypes.fp8_e8m0)

    def steps(self, inputs: TpMoeInputs):
        plan = self.plans(inputs.global_tokens)
        w_all, i_all = self.gather.route(inputs)
        sorted_ret = sort_routes(self.shape, w_all, i_all, plan)
        if plan.prequant:
            a1, a1_scale = self._payload_ag(
                *self.quant(inputs.x_local, quant_dtype=AQ_DTYPE), inputs
            )
        else:
            a1, a1_scale = self.gather(inputs)[0], None
        return plan, w_all, i_all, sorted_ret, a1, a1_scale

    def __call__(self, inputs: TpMoeInputs) -> torch.Tensor:
        plan, w_all, i_all, sorted_ret, a1, a1_scale = self.steps(inputs)
        partial = self.gemms(a1, a1_scale, w_all, i_all, sorted_ret, plan).contiguous()
        return self.comm.reduce_scatter(partial, out=self._y[: inputs.local_tokens])

    def warmup(self, x, wts, ids) -> None:
        plan = self.plans(int(x.shape[0]))
        sorted_ret = sort_routes(self.shape, wts, ids, plan)
        a1, a1_scale = (
            self.quant(x, quant_dtype=AQ_DTYPE) if plan.prequant else (x, None)
        )
        self.gemms(a1, a1_scale, wts, ids, sorted_ret, plan)


class SplitTpMoeAR(SplitTpMoe):

    def __init__(self, weights, ctx, max_local_tokens, comm):
        super().__init__(weights, ctx, max_local_tokens, comm)
        total = max_local_tokens * weights.tp_size
        H = weights.shape.model_dim
        self._x = torch.empty((total, H), dtype=dtypes.bf16, device=ctx.device)
        self._yall = torch.empty((total, H), dtype=dtypes.bf16, device=ctx.device)

    def steps(self, inputs: TpMoeInputs):
        g = inputs.global_tokens
        plan = self.plans(g)
        x = inputs.x_local
        if inputs.comm_mode == "ar_ar":
            x = self.comm.all_reduce(x, out=self._x[:g])
        w_all, i_all = inputs.topk_weights_local, inputs.topk_ids_local
        sorted_ret = sort_routes(self.shape, w_all, i_all, plan)
        a1, a1_scale = (
            self.quant(x, quant_dtype=AQ_DTYPE) if plan.prequant else (x, None)
        )
        return plan, w_all, i_all, sorted_ret, a1, a1_scale

    def __call__(self, inputs: TpMoeInputs) -> torch.Tensor:
        plan, w_all, i_all, sorted_ret, a1, a1_scale = self.steps(inputs)
        partial = self.gemms(a1, a1_scale, w_all, i_all, sorted_ret, plan)
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
    if inputs.comm_mode in ("ar", "ar_ar"):
        xs = inputs.x_local.float()
        if inputs.comm_mode == "ar_ar":
            dist.all_reduce(xs)
        x_all, w_all, i_all = (
            xs.to(dtypes.bf16),
            inputs.topk_weights_local,
            inputs.topk_ids_local,
        )
    else:
        x_all, w_all, i_all = (t.clone() for t in gather(inputs))
    acc = torch_partial(weights, x_all, w_all, i_all, inputs.global_tokens).float()
    dist.all_reduce(acc)
    if inputs.comm_mode in ("ar", "ar_ar"):
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
    """Every rank agreed to skip the current case."""


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
            f" | split={row.get('split_graph_us', nan):.1f} fused={row.get('fused_graph_us', nan):.1f} "
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
        except Exception as exc:  # noqa: BLE001 - warmup must never be fatal
            logger.warning("[jit-warmup] %s M=%d: %s", shape.name, global_tokens, exc)
            ok = False
        torch.cuda.empty_cache()
    return all_ranks_ok(ok, ctx.device)


def _graph_capture_ctx():
    try:
        from aiter.dist.parallel_state import get_tp_group, graph_capture

        get_tp_group()
    except Exception:  # noqa: BLE001 - no aiter parallel state -> plain capture
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
        return us, out.clone()
    finally:
        # a live graph while the next capture warms up is fatal
        graph = out = None
        gc.collect()
        torch.cuda.synchronize()


def run_case(shape, weights, ctx, args, global_tokens, max_local_tokens) -> dict:
    tp = args.tp
    ar = args.comm_mode in ("ar", "ar_ar")
    inputs = make_inputs(
        shape, ctx, tp, global_tokens, args.seed, args.route, args.comm_mode
    )
    moe = (SplitTpMoeAR if ar else SplitTpMoe)(
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
        if rep is not None:
            gate(
                row,
                "split_graph_rel_l2",
                rel_l2(rep, y),
                args.rtol,
                "split graph replay",
            )
        if fused is not None:
            row["fused_graph_us"], rep = _time_graph(fused, inputs, args, ctx)
            if rep is not None:
                gate(
                    row,
                    "fused_graph_rel_l2",
                    rel_l2(rep, fused(inputs)),
                    args.rtol,
                    "fused graph replay",
                )
            set_speedup(row)
    row["comm"] = args.comm_backend.describe()
    return row


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
        return 0
    finally:
        cleanup_dist()


def _sp_rel_l2(actual, expected) -> float:
    err = sum(
        float(((a.float() - e.float().to(a.device)) ** 2).sum())
        for a, e in zip(actual, expected)
    )
    ref = sum(float((e.float() ** 2).sum()) for e in expected)
    return float("nan") if ref == 0.0 else (err / ref) ** 0.5


def _sp_run(devices, fn):
    out, err = [None] * len(devices), [None] * len(devices)

    def work(r):
        try:
            torch.cuda.set_device(devices[r])
            out[r] = fn(r)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            err[r] = exc

    threads = [threading.Thread(target=work, args=(r,)) for r in range(len(devices))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for e in err:
        if e is not None:
            raise e
    for d in devices:
        torch.cuda.synchronize(d)
    return out


def _sp_time_graph(devices, fn, args) -> tuple[float, list | None]:
    graphs = []
    try:
        for _ in range(args.sp_warmup):
            _sp_run(devices, fn)
        streams = [torch.cuda.Stream(d) for d in devices]
        for _ in range(3):
            for r, d in enumerate(devices):
                with torch.cuda.device(d), torch.cuda.stream(streams[r]):
                    fn(r)
            for d in devices:
                torch.cuda.synchronize(d)
        outs = []
        for r, d in enumerate(devices):
            with torch.cuda.device(d):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=streams[r]):
                    outs.append(fn(r))
                graphs.append(g)
        for d in devices:
            torch.cuda.synchronize(d)
        best = float("inf")
        for _ in range(args.rounds + 1):
            evs = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in devices
            ]
            gate_ = threading.Barrier(len(devices))

            def replay(r, evs=evs, gate_=gate_):
                torch.cuda.set_device(devices[r])
                with torch.cuda.stream(streams[r]):
                    gate_.wait()
                    evs[r][0].record(streams[r])
                    for _ in range(args.iters):
                        graphs[r].replay()
                    evs[r][1].record(streams[r])

            ts = [
                threading.Thread(target=replay, args=(r,)) for r in range(len(devices))
            ]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            for d in devices:
                torch.cuda.synchronize(d)
            best = min(
                best, max(e[0].elapsed_time(e[1]) for e in evs) * 1000.0 / args.iters
            )
        for r, d in enumerate(devices):
            with torch.cuda.device(d), torch.cuda.stream(streams[r]):
                graphs[r].replay()
        for d in devices:
            torch.cuda.synchronize(d)
        return best, [o.clone() for o in outs]
    finally:
        graphs = None
        gc.collect()
        for d in devices:
            torch.cuda.synchronize(d)


def _sp_reference(weights, ctxs, inputs, global_tokens):
    dev0 = ctxs[0].device
    if inputs[0].comm_mode in ("ar", "ar_ar"):
        if inputs[0].comm_mode == "ar_ar":
            x_all = sum(i.x_local.float().to(dev0) for i in inputs).to(dtypes.bf16)
        else:
            x_all = inputs[0].x_local.to(dev0)
        w_all, i_all = inputs[0].topk_weights_local.to(dev0), inputs[
            0
        ].topk_ids_local.to(dev0)
    else:
        x_all = torch.cat([i.x_local.to(dev0) for i in inputs])
        w_all = torch.cat([i.topk_weights_local.to(dev0) for i in inputs])
        i_all = torch.cat([i.topk_ids_local.to(dev0) for i in inputs])
    total = None
    for r, c in enumerate(ctxs):
        with torch.cuda.device(c.device):
            p = torch_partial(
                weights[r],
                x_all.to(c.device),
                w_all.to(c.device),
                i_all.to(c.device),
                global_tokens,
            )
        p = p.float().to(dev0)
        total = p if total is None else total + p
    if inputs[0].comm_mode in ("ar", "ar_ar"):
        return [total] * len(ctxs)
    m = inputs[0].local_tokens
    return [total[r * m : (r + 1) * m] for r in range(len(ctxs))]


def run_case_sp(
    shape, weights, ctxs, args, global_tokens, max_local_tokens, p2p, group
) -> dict:
    devices = [c.device for c in ctxs]
    inputs = [
        make_inputs(
            shape, c, args.tp, global_tokens, args.seed, args.route, args.comm_mode
        )
        for c in ctxs
    ]
    split_cls = SplitTpMoeAR if args.comm_mode in ("ar", "ar_ar") else SplitTpMoe
    split = []
    for r, c in enumerate(ctxs):
        with torch.cuda.device(c.device):
            split.append(split_cls(weights[r], c, max_local_tokens, p2p.comm(r)))
            x, ids, wts = _warmup_inputs(shape, global_tokens, c.device)
            try:
                split[r].warmup(x, wts, ids)
            except Exception as exc:
                raise SkipCase(f"kernel build failed: {exc}") from exc
    row = new_row(
        shape,
        args,
        global_tokens,
        inputs[0].local_tokens,
        weights[0].local_inter_dim,
        split[0].kernel_names(global_tokens),
        "p2p-single-process",
    )
    y = [t.clone() for t in _sp_run(devices, lambda r: split[r](inputs[r]))]
    if any(bool(torch.isnan(t).any()) for t in y):
        raise AssertionError(
            f"{shape.tag(args.tp)} tokens={global_tokens}: output has NaN"
        )
    expected = None
    if global_tokens <= args.accuracy_max_tokens:
        expected = _sp_reference(weights, ctxs, inputs, global_tokens)
        gate(row, "rel_l2", _sp_rel_l2(y, expected), args.rtol, "split vs torch")

    fused = None
    if args.impl in ("fused", "both"):
        fused = []
        for r, c in enumerate(ctxs):
            with torch.cuda.device(c.device):
                fused.append(
                    MegaMoeTP(
                        weights[r],
                        c,
                        max_local_tokens,
                        group=group,
                        comm_mode=args.comm_mode,
                        comm_dtype=args.comm_dtype,
                    )
                )
        _sp_run(devices, lambda r: fused[r](inputs[r]))
        for f in fused:
            f.engine.engine.clear_errors()
        yf = [t.clone() for t in _sp_run(devices, lambda r: fused[r](inputs[r]))]
        errs = [f.engine.poll_errors() for f in fused]
        if any(errs):
            raise AssertionError(
                f"{shape.tag(args.tp)} tokens={global_tokens}: fused watchdog fired {errs}"
            )
        gate(row, "fused_rel_l2", _sp_rel_l2(yf, y), args.fused_rtol, "fused vs split")
        if expected is not None:
            gate(
                row,
                "fused_ref_rel_l2",
                _sp_rel_l2(yf, expected),
                args.fused_ref_rtol,
                "fused vs torch",
            )
            masked = _masked_inputs(inputs, shape.experts)
            ym = _sp_run(devices, lambda r: fused[r](masked[r][0]))
            errs = [f.engine.poll_errors() for f in fused]
            if any(errs):
                raise AssertionError(f"masked ids: fused watchdog fired {errs}")
            gate(
                row,
                "fused_masked_rel_l2",
                _sp_rel_l2(
                    ym,
                    _sp_reference(weights, ctxs, [m for _, m in masked], global_tokens),
                ),
                args.fused_ref_rtol,
                "fused (masked ids) vs torch",
            )
        _sp_empty_batch(fused, devices, shape, args.comm_mode)
    if not args.no_perf:
        row["ag_wire"] = split[0].plans(global_tokens).ag_wire
        row["split_graph_us"], rep = float("nan"), None
        if args.impl != "fused":
            row["split_graph_us"], rep = _sp_time_graph(
                devices, lambda r: split[r](inputs[r]), args
            )
        if rep is not None:
            gate(
                row,
                "split_graph_rel_l2",
                _sp_rel_l2(rep, y),
                args.rtol,
                "split graph replay",
            )
        if fused is not None:
            row["fused_graph_us"], rep = _sp_time_graph(
                devices, lambda r: fused[r](inputs[r]), args
            )
            if rep is not None:
                gate(
                    row,
                    "fused_graph_rel_l2",
                    _sp_rel_l2(rep, yf),
                    args.rtol,
                    "fused graph replay",
                )
            set_speedup(row)
    return row


def _masked_inputs(inputs, experts):
    """(ids with ~20% set to -1 / >= experts, the same with those routes as
    expert 0 at weight 0 for the reference) per rank."""
    out = []
    for i in inputs:
        g = torch.Generator(device=i.topk_ids_local.device).manual_seed(77)
        sel = torch.rand(
            i.topk_ids_local.shape, device=i.topk_ids_local.device, generator=g
        )
        ids = torch.where(sel < 0.1, -1, i.topk_ids_local)
        ids = torch.where((sel >= 0.1) & (sel < 0.2), experts + 3, ids)
        drop = sel < 0.2
        w0 = torch.where(drop, 0.0, i.topk_weights_local)
        ref_ids = torch.where(drop, 0, i.topk_ids_local)
        meta = torch.cat([ref_ids, w0.view(torch.int32)], 1).to(torch.int32)
        out.append(
            (
                replace(i, topk_ids_local=ids.to(i.topk_ids_local.dtype)),
                replace(
                    i,
                    topk_ids_local=ref_ids,
                    topk_weights_local=w0,
                    route_meta_local=meta,
                ),
            )
        )
    return out


def _sp_empty_batch(fused, devices, shape, comm_mode) -> None:
    def run(r):
        x = torch.empty((0, shape.model_dim), dtype=dtypes.bf16, device=devices[r])
        ids = torch.empty((0, shape.topk), dtype=torch.int32, device=devices[r])
        w = torch.empty((0, shape.topk), dtype=torch.float32, device=devices[r])
        return fused[r].engine(x, w, ids)

    ys = _sp_run(devices, run)
    if any(tuple(y.shape) != (0, shape.model_dim) for y in ys):
        raise AssertionError(f"{comm_mode} empty batch: {[y.shape for y in ys]}")


def _flydsl_multi_device() -> None:
    from p2p_collectives import flydsl_multi_device

    flydsl_multi_device()


def main_single_process(args) -> int:
    from p2p_collectives import P2PGroup, PeerArenaGroup

    _flydsl_multi_device()

    args.tp = args.tp or torch.cuda.device_count()
    devices = [torch.device("cuda", i) for i in range(args.tp)]
    torch.cuda.set_device(devices[0])
    gfx = get_gfx()
    if gfx not in SUPPORTED_GFX:
        print(f"[SKIP] a4w4 MoE needs {SUPPORTED_GFX}, found {gfx}")
        return 0
    tokens, max_local = _check_tokens(args)
    group = PeerArenaGroup(devices)
    p2p = P2PGroup(devices, max_local)
    ctxs = [DistCtx(rank=r, world=args.tp, device=d) for r, d in enumerate(devices)]
    print(
        f"[ENV] gfx={gfx} cu={get_cu_num()} tp={args.tp} quant=a4w4 mode=single-process "
        f"comm_mode={args.comm_mode} baseline_comm=p2p fmoe_csv={AITER_CONFIGS.AITER_CONFIG_FMOE_FILE}",
        flush=True,
    )
    rows: list[dict] = []
    for name in args.models:
        shape = MODELS[name]
        print(f"\n=== {shape.tag(args.tp)} ===", flush=True)
        weights = []
        for c in ctxs:
            with torch.cuda.device(c.device):
                weights.append(build_sharded_weights(shape, c, args.tp, args.seed))
        for global_tokens in tokens:
            try:
                row = run_case_sp(
                    shape, weights, ctxs, args, global_tokens, max_local, p2p, group
                )
            except SkipCase as exc:
                print(f"[SKIP] {shape.name} M={global_tokens}: {exc}", flush=True)
                continue
            rows.append(row)
            print_row(row, args.no_perf)
            for d in devices:
                with torch.cuda.device(d):
                    torch.cuda.empty_cache()
        del weights
    write_table(rows, args, "TP MoE, a4w4, single process")
    return 0


def _check_tokens(args):
    bad = [t for t in args.tokens if t <= 0 or t % args.tp]
    if bad:
        raise ValueError(f"tokens {bad} must be positive multiples of TP={args.tp}")
    tokens = sorted(set(args.tokens))
    return tokens, max(tokens) // args.tp


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
    )
    p.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    p.add_argument(
        "--tokens",
        type=int,
        nargs="*",
        default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        help="GLOBAL token counts (multiples of --tp).",
    )
    p.add_argument(
        "--tp", type=int, default=0, help="TP size; 0 = WORLD_SIZE (or every GPU)."
    )
    p.add_argument(
        "--comm-mode",
        choices=["ag_rs", "ar", "ar_ar"],
        default="ag_rs",
        help="ag_rs: AllGather before, ReduceScatter after; ar: AllReduce after "
        "(input replicated); ar_ar: AllReduce of input partials and output.",
    )
    p.add_argument("--impl", choices=["unfused", "fused", "both"], default="both")
    p.add_argument(
        "--comm-dtype",
        choices=["fp8", "bf16"],
        default="fp8",
        help="fused: dtype of the reduce partials and route rows.",
    )
    p.add_argument(
        "--route",
        choices=["random", "balanced"],
        default="random",
        help="random: top-k of random scores; balanced: every expert equally loaded.",
    )
    p.add_argument(
        "--single-process",
        action="store_true",
        help="One process drives all --tp GPUs through peer access (no torchrun).",
    )
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--rounds", type=int, default=3, help="Timed rounds; the fastest is kept."
    )
    p.add_argument(
        "--capture-warmup",
        type=int,
        default=400,
        help="torchrun: eager calls before a graph capture (see _time_graph).",
    )
    p.add_argument(
        "--sp-warmup",
        type=int,
        default=20,
        help="single-process: eager calls before a graph capture.",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--accuracy-max-tokens",
        type=int,
        default=128,
        help="Skip the (slow) torch reference above this global token count.",
    )
    p.add_argument(
        "--rtol", type=float, default=0.06, help="rel_l2 gate vs torch / replay."
    )
    p.add_argument(
        "--fused-rtol",
        type=float,
        default=0.08,
        help="rel_l2 gate fused vs split (each carries its own FP4/FP8 error).",
    )
    p.add_argument(
        "--fused-ref-rtol",
        type=float,
        default=0.0,
        help="rel_l2 gate fused vs torch; 0: 0.045 (--comm-dtype fp8) / 0.01 (bf16).",
    )
    p.add_argument("--no-perf", action="store_true", help="Accuracy only.")
    p.add_argument("--csv", default=None, help="Write the result table here.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.fused_ref_rtol:
        args.fused_ref_rtol = 0.045 if args.comm_dtype == "fp8" else 0.01
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.single_process:
        return main_single_process(args)
    return main_torchrun(args)


if __name__ == "__main__":
    sys.exit(main())
