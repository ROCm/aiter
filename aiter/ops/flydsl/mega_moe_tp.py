# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused tensor-parallel MoE layer: AllGather + GEMM1 + GEMM2 + ReduceScatter.

Parallelization model
---------------------
Experts are *replicated* across the TP group and only ``inter_dim`` is sharded,
so rank ``r`` owns ``w1[E, 2*I_r, H]`` / ``w2[E, H, I_r]`` with ``I_r =
inter_dim // TP``.  Activations arrive sequence-parallel: rank ``r`` holds the
global token rows ``[r*m, (r+1)*m)`` with ``m = M // TP``.  One layer is::

    (1) AG  : x_all[M, H]     = AllGather_token(x_local[m, H])
    (2) G1  : h[M*topk, I_r]  = act(gather(x_all) @ w1_r^T)
    (3) G2  : p[M*topk, H]    = h @ w2_r^T
    (4) RD  : q[M, H]         = sum_k topk_weight[t,k] * p[t,k]
    (5) RS  : y_local[m, H]   = sum_over_ranks(q)[r*m : (r+1)*m]

Steps (2)-(4) are exactly the local two-stage MoE that ``test_moe_2stage.py``
benchmarks with ``-dim H,I_r``; this module keeps those kernels untouched and
fuses the work around them.

What is fused here
------------------
``quantize -> AllGather``
    The MXFP4 activation quantization that GEMM1 would do anyway is hoisted in
    front of the collective, so the wire row shrinks from ``H*2`` bytes to
    ``H/2 + H/32`` (3.77x) and each rank quantizes only its own ``m`` rows
    instead of all ``M``.  Per-1x32 MX quantization is row-local, so this is
    bit-identical to quantizing after the AllGather.

    The hoist needs a GEMM1 that accepts a pre-quantized FP4 operand.  In the
    ``flydsl_mxmoe_g1_a4w4_*`` family that is every non-``f16in`` variant
    (``MXFP4_G1_VARIANTS``); the ``f16in`` ones read raw BF16 and quantize
    inline.  Small ``M`` tunes onto ``BM=16``, which has *no* pre-quantized
    variant compiled at all -- ``MXFP4_G1_VARIANTS["fp4"]`` holds only
    ``(16, True, True)``, because ``native_scale_layout_for(16, "fp4")`` puts
    BM16 on a different GEMM1/GEMM2 scale-layout contract.

    So to keep small ``M`` on the FP4 wire, ``_resolve_plan`` borrows the tuned
    row of the first larger token bucket whose GEMM1 *does* take a
    pre-quantized operand, and substitutes that whole row (stage1 *and* stage2,
    so the scale-layout contract between them stays intact).  The GEMM is then
    tile-tuned for more tokens than the call actually has -- a legal kernel for
    the shape, but not the tuned-optimal one.  ``ag_wire='bf16'`` opts out and
    keeps the tuned inline-quant row.

``routing metadata AllGather``
    ``topk_ids`` and ``topk_weights`` are packed into one int32 payload so the
    routing costs a single collective rather than two.

Only a4w4 (MXFP4 activation x MXFP4 weight, ``QuantType.per_1x32``) is wired up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.ops.quant import get_hip_quant

from .kernels.mega_moe_tp.collectives import TpMoeCollectives
from .moe_common import GateMode
from .mxfp4_kname import (
    MXFP4_G1_VARIANTS,
    _is_mxfp4_kname,
    _parse_mxfp4_g1_kname,
)

logger = logging.getLogger("aiter")

__all__ = ["MegaMoeTP", "MegaMoeTPConfig", "mega_moe_tp_supported"]

QUANT_TYPE = QuantType.per_1x32
AQ_DTYPE = dtypes.fp4x2
WQ_DTYPE = dtypes.fp4x2

#: Wire formats understood by :class:`MegaMoeTP` for the AllGather leg.
#:
#: ``auto`` prefers ``fp4_1x32`` at every ``M``, substituting a pre-quantized
#: GEMM1 from a larger token bucket when the tuned row for this ``M`` quantizes
#: inline; it only falls back to ``bf16`` when no such row exists anywhere.
#: ``fp4_1x32`` is the same but raises instead of falling back.  ``bf16`` pins
#: the BF16 wire and always keeps the tuned row.
AG_WIRE_MODES = ("auto", "fp4_1x32", "bf16")
#: Wire formats understood for the ReduceScatter leg.
RS_WIRE_MODES = ("auto", "bf16")


@dataclass(frozen=True)
class MegaMoeTPConfig:
    """Everything that is fixed for the lifetime of one fused TP MoE layer."""

    rank: int
    world_size: int
    model_dim: int
    inter_dim: int  # per-rank shard, i.e. inter_dim_full // world_size
    experts: int
    topk: int
    max_local_tokens: int
    activation: ActivationType = ActivationType.Situv2
    beta: float | None = None
    linear_beta: float | None = None
    ag_wire: str = "auto"
    rs_wire: str = "auto"

    def __post_init__(self):
        if self.world_size <= 0:
            raise ValueError(f"world_size must be positive, got {self.world_size}")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank {self.rank} outside world {self.world_size}")
        if self.model_dim % 32:
            raise ValueError(
                "model_dim must be a multiple of the 32-wide MX group, got "
                f"{self.model_dim}"
            )
        if self.max_local_tokens <= 0:
            raise ValueError(
                f"max_local_tokens must be positive, got {self.max_local_tokens}"
            )
        if self.ag_wire not in AG_WIRE_MODES:
            raise ValueError(f"ag_wire must be one of {AG_WIRE_MODES}")
        if self.rs_wire not in RS_WIRE_MODES:
            raise ValueError(f"rs_wire must be one of {RS_WIRE_MODES}")

    @property
    def max_global_tokens(self) -> int:
        return self.max_local_tokens * self.world_size


def mega_moe_tp_supported(gfx: str | None = None) -> bool:
    """Whether the fused TP MoE has a kernel path on this device."""
    if gfx is None:
        from aiter.jit.utils.chip_info import get_gfx

        gfx = get_gfx()
    return gfx == "gfx950"


# ---------------------------------------------------------------------------
# GEMM1 capability probe
# ---------------------------------------------------------------------------
def _gemm1_takes_prequantized_fp4(kernel_name1: str) -> bool:
    """Can this GEMM1 consume an already-MXFP4-quantized A operand?

    ``flydsl_mxmoe_g1_a4w4_*`` splits into two families:

    * ``_f16in`` -- inline quant.  The kernel reads BF16 ``hidden_states`` and
      ignores the packed-A/scale buffers entirely; handing it FP4 faults.
    * everything else -- A arrives packed FP4 plus a sorted E8M0 scale.

    ``flydsl_moe1_afp4_wfp4_bf16_*`` (the other a4w4 GEMM1 port) is always
    pre-quantized, which is why it is accepted here without a name parse.
    """
    if not isinstance(kernel_name1, str) or not kernel_name1:
        return False
    if kernel_name1.startswith("flydsl_moe1_afp4_wfp4_"):
        return True
    if not _is_mxfp4_kname(kernel_name1):
        return False
    try:
        parsed = _parse_mxfp4_g1_kname(kernel_name1)
    except ValueError:
        return False
    if parsed["a_dtype"] != "fp4" or parsed["inline_quant"]:
        return False
    variant = (parsed["BM"], parsed["use_nt"], False)
    return variant in MXFP4_G1_VARIANTS["fp4"]


def _partial_keyword(fn, *keys: str) -> str:
    """Dig the first matching keyword out of a (possibly nested) functools.partial.

    The MXFP4 port stores the GEMM name under ``kernelName1``/``kernelName2``
    while the FlyDSL stage wrappers use a plain ``kernelName``.
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
# Runtime
# ---------------------------------------------------------------------------
class MegaMoeTP:
    """Fused AllGather + GEMM1 + GEMM2 + ReduceScatter MoE for one TP rank.

    ``w1``/``w2`` are this rank's ``inter_dim`` shard in the layout the tuned
    a4w4 kernels expect (``shuffle_weight(..., layout=(16, 16))`` for the packed
    FP4 payload and ``fp4_utils.e8m0_shuffle`` for the scales).  ``forward``
    takes the sequence-parallel activation shard and returns this rank's shard
    of the layer output, so the object is a drop-in for the unfused
    AllGather/MoE/ReduceScatter chain.

    Every buffer is allocated in ``__init__``; ``forward`` is allocation-free
    apart from what the MoE kernels allocate internally.
    """

    name = "mega_moe_tp"

    def __init__(
        self,
        config: MegaMoeTPConfig,
        *,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        group=None,
        device: torch.device | None = None,
    ):
        self.cfg = config
        self.group = group
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.w1, self.w1_scale = w1, w1_scale
        self.w2, self.w2_scale = w2, w2_scale

        cfg = self.cfg
        if w2.shape[1] != cfg.model_dim:
            raise ValueError(
                f"w2 model_dim {w2.shape[1]} disagrees with config {cfg.model_dim}"
            )
        if w1.shape[0] != cfg.experts:
            raise ValueError(
                f"w1 expert count {w1.shape[0]} disagrees with config {cfg.experts}"
            )

        # The collective arenas are built on first use: which wire a shape needs
        # follows from its tuned GEMM1, and holding both would cost the sum of
        # a BF16 and an MXFP4 staging buffer for nothing. The choice is
        # deterministic, so every rank reaches the same arena at the same point
        # and the IPC exchange inside stays collective-safe.
        self._comm: dict[str, TpMoeCollectives] = {}

        self._quant = get_hip_quant(QUANT_TYPE)
        situ = cfg.activation == ActivationType.Situv2
        # Exactly the kwargs the ordinary two-stage path uses, so the tuned
        # config lookup lands on the same row as an unfused call would.
        self._moe_kwargs = {
            "w1_scale": w1_scale,
            "w2_scale": w2_scale,
            "quant_type": QUANT_TYPE,
            "activation": cfg.activation,
            "doweight_stage1": False,
            "intermediate_pad": 0,
            "hidden_pad": 0,
            "bias1": None,
            "bias2": None,
            "swiglu_limit": None,
            "beta": cfg.beta if situ else None,
            "linear_beta": cfg.linear_beta if situ else None,
            "gate_mode": GateMode.SEPARATED.value,
        }
        self._plan_cache: dict[int, _CasePlan] = {}

    # -- plan ---------------------------------------------------------------
    def plan(self, global_tokens: int) -> "_CasePlan":
        """Resolve (and cache) the wire format and kernel names for one M."""
        from aiter.fused_moe import get_padded_M

        bucket = int(get_padded_M(global_tokens))
        cached = self._plan_cache.get(bucket)
        if cached is not None:
            return cached
        plan = self._resolve_plan(bucket)
        self._plan_cache[bucket] = plan
        return plan

    def _tuned_row(self, bucket: int):
        """Look up the tuned two-stage config for one token bucket.

        Returns ``(metadata, kernel1, kernel2)``.  A failed lookup yields
        ``(None, "", "")`` rather than raising: the caller either falls back to
        the BF16 wire or moves on to the next bucket in the probe.
        """
        from aiter.fused_moe import get_2stage_cfgs
        from aiter.ops.flydsl.moe_common import GateMode

        cfg = self.cfg
        try:
            metadata = get_2stage_cfgs(
                bucket,
                cfg.model_dim,
                cfg.inter_dim,
                cfg.experts,
                cfg.topk,
                dtypes.bf16,
                AQ_DTYPE,
                WQ_DTYPE,
                QUANT_TYPE,
                True,  # use_g1u1
                cfg.activation,
                False,  # doweight_stage1
                0,
                0,
                True,  # is_shuffled
                GateMode.SEPARATED.value,
            )
        except Exception as exc:  # noqa: BLE001 - probe only, never fatal
            logger.warning(
                "[mega_moe_tp] could not resolve tuned kernels for M=%d: %s",
                bucket,
                exc,
            )
            return None, "", ""
        return (
            metadata,
            _partial_keyword(metadata.stage1, "kernelName1", "kernelName"),
            _partial_keyword(metadata.stage2, "kernelName2", "kernelName"),
        )

    def _probe_prequant_row(self, bucket: int):
        """First bucket above ``bucket`` whose tuned GEMM1 reads pre-quantized A.

        Buckets are powers of two (:func:`get_padded_M`), and above
        ``_PADDED_M_TIERS[0]`` the tuner reuses that top row, so the walk stops
        there instead of asking for rows the CSV does not carry.
        """
        from aiter.fused_moe import _PADDED_M_TIERS

        top = int(_PADDED_M_TIERS[0])
        probe = bucket * 2
        while probe <= top:
            metadata, kernel1, kernel2 = self._tuned_row(probe)
            if _gemm1_takes_prequantized_fp4(kernel1):
                return probe, metadata, kernel1, kernel2
            probe *= 2
        return None

    def _resolve_plan(self, bucket: int) -> "_CasePlan":
        cfg = self.cfg
        _, kernel1, kernel2 = self._tuned_row(bucket)
        prequant_ok = _gemm1_takes_prequantized_fp4(kernel1)

        if cfg.ag_wire == "bf16":
            return _CasePlan(bucket, "bf16", kernel1, kernel2, None, bucket)
        if prequant_ok:
            return _CasePlan(bucket, "fp4_1x32", kernel1, kernel2, None, bucket)

        # The tuned row for this bucket quantizes inline, so it cannot read the
        # FP4 wire, and at BM16 no pre-quantized variant exists to retune onto.
        # Borrow a larger bucket's row instead -- both stages together, so the
        # GEMM1/GEMM2 scale-layout contract stays self-consistent.
        borrowed = self._probe_prequant_row(bucket)
        if borrowed is not None:
            src, metadata, sub1, sub2 = borrowed
            logger.debug(
                "[mega_moe_tp] M=%d tunes onto inline-quant %r; borrowing the "
                "M=%d row (%r) to stay on the FP4 wire",
                bucket,
                kernel1,
                src,
                sub1,
            )
            return _CasePlan(bucket, "fp4_1x32", sub1, sub2, metadata, src)

        if cfg.ag_wire == "fp4_1x32":
            raise ValueError(
                f"ag_wire='fp4_1x32' needs a pre-quantized GEMM1, but M={bucket} "
                f"resolves to {kernel1!r}, which quantizes inline, and no larger "
                "token bucket for this shape resolves to one either. Use "
                "ag_wire='bf16' or tune a non-f16in GEMM1 for this shape."
            )
        return _CasePlan(bucket, "bf16", kernel1, kernel2, None, bucket)

    # -- stages -------------------------------------------------------------
    def _collectives(self, wire: str) -> TpMoeCollectives:
        comm = self._comm.get(wire)
        if comm is None:
            cfg = self.cfg
            comm = TpMoeCollectives(
                rank=cfg.rank,
                world_size=cfg.world_size,
                model_dim=cfg.model_dim,
                topk=cfg.topk,
                max_local_tokens=cfg.max_local_tokens,
                device=self.device,
                group=self.group,
                fp4_wire=wire == "fp4_1x32",
            )
            self._comm[wire] = comm
        return comm

    def all_gather(self, x_local: torch.Tensor, topk_weights, topk_ids, plan=None):
        """Gather the activation shard (and the route) into the global token set.

        One P2P push kernel moves all of it: the activation payload, its E8M0
        scales on the MXFP4 wire, and the routing ids/weights. Returns
        ``(a1, a1_scale, topk_weights_all, topk_ids_all)`` where ``a1`` is BF16
        ``[M, H]`` on the BF16 wire and packed FP4 ``[M, H/2]`` with a matching
        E8M0 ``[M, H/32]`` scale on the MXFP4 wire.
        """
        cfg = self.cfg
        m = int(x_local.shape[0])
        total = m * cfg.world_size
        if m > cfg.max_local_tokens:
            raise ValueError(
                f"local tokens {m} exceeds max_local_tokens {cfg.max_local_tokens}"
            )
        plan = plan or self.plan(total)
        comm = self._collectives(plan.ag_wire)

        if plan.ag_wire == "fp4_1x32":
            # Quantize before the push: the wire row drops from H*2 to
            # H/2 + H/32 bytes and each rank only quantizes its own m rows.
            # Per-1x32 MX quant is row-local, so this matches quantizing the
            # gathered tensor exactly.
            payload, scale = self._quant(x_local, quant_dtype=AQ_DTYPE)
            payload = payload.view(torch.uint8)
            scale = scale.view(torch.uint8)
        else:
            payload, scale = x_local, None

        gathered = comm.all_gather(payload, scale, topk_ids, topk_weights)
        if plan.ag_wire == "fp4_1x32":
            a1 = gathered.payload.view(AQ_DTYPE)
            a1_scale = gathered.scale.view(dtypes.fp8_e8m0)
        else:
            a1 = gathered.payload
            a1_scale = None
        return a1, a1_scale, gathered.topk_weights, gathered.topk_ids

    def local_moe(self, a1, a1_scale, topk_weights_all, topk_ids_all, plan=None):
        """Run GEMM1 + activation + GEMM2 + weighted top-k reduce for all tokens.

        The result lands directly in the symmetric arena, so the ReduceScatter
        that follows reads it in place instead of staging a copy.  GEMM2's
        atomic epilogue needs a zeroed target and ``moe_sorting`` zeroes
        whatever output buffer it is handed, so the arena slice is safe to reuse
        every call.

        The BF16 wire hands the activation to the ordinary public entry point;
        only the MXFP4 wire needs the private one, to force the pre-quantized
        activation path (``fused_moe`` exposes no hook for that).
        """
        total = int(topk_ids_all.shape[0])
        plan = plan or self.plan(total)
        output = self._collectives(plan.ag_wire).partial_buffer(total)
        if a1_scale is None:
            return fused_moe(
                a1,
                self.w1,
                self.w2,
                topk_weights_all,
                topk_ids_all,
                output=output,
                **self._moe_kwargs,
            )

        from aiter.fused_moe import _fused_moe_impl

        kwargs = dict(self._moe_kwargs)
        kwargs["quant_type"] = kwargs["quant_type"].value
        kwargs["activation"] = kwargs["activation"].value
        # The activation arrives packed FP4, so the output dtype cannot be
        # inferred from it the way the BF16 wire allows.
        kwargs["dtype"] = dtypes.bf16
        if plan.metadata is None:
            transform = _prequant_transform
        else:
            # This bucket's own tuned row quantizes inline; run the borrowed
            # row instead of whatever the lookup inside _fused_moe_impl finds.
            borrowed = replace(plan.metadata, prequant=True)
            transform = lambda _metadata: borrowed  # noqa: E731
        return _fused_moe_impl(
            a1,
            self.w1,
            self.w2,
            topk_weights_all,
            topk_ids_all,
            a1_scale=a1_scale,
            output=output,
            _q_dtype_a=AQ_DTYPE,
            _metadata_transform=transform,
            **kwargs,
        )

    def reduce_scatter(self, local_tokens: int, plan: "_CasePlan"):
        """Sum the per-rank partials across TP and keep this rank's token shard."""
        return self._collectives(plan.ag_wire).reduce_scatter(local_tokens)

    # -- public entry point -------------------------------------------------
    def forward(self, x_local, topk_weights, topk_ids) -> torch.Tensor:
        cfg = self.cfg
        m = int(x_local.shape[0])
        if x_local.dtype != dtypes.bf16:
            raise TypeError(f"x_local must be bfloat16, got {x_local.dtype}")
        if x_local.shape[1] != cfg.model_dim:
            raise ValueError(
                f"x_local model_dim {x_local.shape[1]} != {cfg.model_dim}"
            )
        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)
        if topk_weights.dtype != torch.float32:
            topk_weights = topk_weights.to(torch.float32)
        if not x_local.is_contiguous():
            x_local = x_local.contiguous()
        plan = self.plan(m * cfg.world_size)
        a1, a1_scale, wts, ids = self.all_gather(
            x_local, topk_weights, topk_ids, plan=plan
        )
        self.local_moe(a1, a1_scale, wts, ids, plan=plan)
        return self.reduce_scatter(m, plan)

    __call__ = forward


@dataclass(frozen=True)
class _CasePlan:
    """The per-token-bucket decisions the runtime caches.

    ``metadata`` is the two-stage config to force, set only when this bucket's
    own tuned row had to be swapped out to stay on the FP4 wire; ``None`` means
    the ordinary lookup already lands on the right row.  ``gemm_bucket`` records
    which bucket the kernels came from, so it is visible in tests and logs when
    it is not ``tokens``.
    """

    tokens: int
    ag_wire: str
    gemm1_kernel: str
    gemm2_kernel: str
    metadata: object | None = None
    gemm_bucket: int = 0


def _prequant_transform(metadata):
    """Force the pre-quantized activation path for an FP4 wire.

    ``_make_mxfp4_metadata`` only sets ``prequant`` for FP8 activations, and the
    caller-side promotion in ``fused_moe_2stages`` skips ``block_m == 16``
    because that is the inline-quant variant.  Here the wire format has already
    been validated against the GEMM1 variant, so the promotion is unconditional.
    """
    return replace(metadata, prequant=True)
