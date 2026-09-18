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

``GEMM2 -> ReduceScatter``
    GEMM2's atomic epilogue already accumulates into the symmetric arena slice
    the ReduceScatter reads, so the only thing between them is "have all peers
    landed".  :mod:`.kernels.mega_moe_tp.stage2_rs` folds the publish and the
    pull into GEMM2's own kernel, paying a device-side counter for that instead
    of two launches and the host gaps around them.  Measured on TP8 kimi3
    against the same run with ``rs_fuse='off'``: 373 -> 336 us at 8 global
    tokens, 386 -> 367 at 64, and level at 512.

    This needs a GEMM2 that accumulates in place, i.e. a non-persistent
    ``atomic`` row; a tuned ``reduce`` row stages per-route output and needs a
    separate reduction kernel afterwards, so those fall back to the standalone
    collective automatically.

Only a4w4 (MXFP4 activation x MXFP4 weight, ``QuantType.per_1x32``) is wired up.
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass, field, replace

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

#: Send the BF16 wire through ``_fused_moe_impl`` so it can reach the fused
#: ReduceScatter. Off by default: a net loss in eager mode, free under graphs.
_BF16_VIA_IMPL = os.environ.get("AITER_TP_MEGA_BF16_VIA_IMPL", "0") == "1"
#: Run GEMM1 + GEMM2 + ReduceScatter as a single kernel where the tuned pair
#: allows it -- which is the BM16 inline-quant GEMM1 the bf16 wire picks at small
#: M, so this is the decode path. Measured TP4, `--ag-wire auto`, e2e:
#: -10.8% / -5.9% / -4.8% on glm5 at 32/64/128 tokens and -5.7% on dsv3 at 32,
#: and neutral (within +-0.3%) everywhere the gate declines. Set to 0 to pin the
#: separate GEMM1 / GEMM2 / ReduceScatter launches.
#: DEFAULT OFF pending a root cause: the merged kernel returns garbage
#: (rel_l2 1.0) for fewer than 8 local rows, while 8/16/32/64/128/256 are all
#: correct. Measured with the wire forced to bf16 so the gate applies:
#:
#:     local rows   2      4      8      16
#:     dsv3         0.003  1.000  0.005  0.006
#:     glm5         1.000  1.000  0.006  0.005
#:
#: The shared RS tail is *not* the culprit -- it is correct at 2 local rows on
#: the fp4 wire (dsv3 TP8 M=16, rel_l2 0.0040) -- so this is specific to the
#: merged kernel. Gating at ">= 8" would be guessing the boundary from four
#: points, which is the mistake already made once with the BM16 gate, so the
#: feature is off until the mechanism is understood.
_STAGE12 = os.environ.get("AITER_TP_MEGA_STAGE12", "0") == "1"
#: Bypass tuned lookup and pin GEMM1/GEMM2 to the two families that expose a
#: ``_composition`` hook -- the prerequisite for one kernel covering every a4w4
#: shape. Costs per-shape tuning; buys a single code path.
_PIN_KERNELS = os.environ.get("AITER_TP_MEGA_PIN_KERNELS", "0") == "1"
#: Sort/tile block for the pinned rows: an int to fix it, or "auto" to pick per
#: token bucket. Must be a pre-quantized fp4 GEMM1 variant, i.e. BM in
#: {32, 64, 128} -- BM16 exists only as inline-quant, which the fp4 wire cannot
#: use. Note BM128 has no ``nt`` variant, so ``use_nt`` follows the table.
_PIN_BM = os.environ.get("AITER_TP_MEGA_PIN_BM", "auto")
#: Largest bucket still served by each block size, in order. Padding is
#: ``experts * (BM - 1)`` sorted rows, so a big BM is pure overhead until the
#: routes per expert catch up; past that the wider tile wins. Measured on TP4
#: across all four models, e2e us, best block size per token count:
#:
#:     M       128  512  1024  2048  4096  8192  32768
#:     winner   32   32    32    64  64/128  128   128
#:
#: 4096 splits 2-2 between 64 and 128, so it is resolved by worst-case cost:
#: picking 128 costs glm5 11.6%, picking 64 costs kimi3 35.7%.
_PIN_BM_LADDER = ((1024, 32), (2048, 64), (1 << 30, 128))


def _pinned_block_m(bucket: int) -> tuple[int, bool]:
    """(block_m, use_nt) for one token bucket on the pinned GEMM1 family."""
    if _PIN_BM != "auto":
        bm = int(_PIN_BM)
    else:
        bm = next(b for limit, b in _PIN_BM_LADDER if bucket <= limit)
    if bm not in (32, 64, 128):
        raise ValueError(f"pinned block_m must be 32/64/128 (fp4 prequant), got {bm}")
    # (BM, use_nt, inline_quant=False) must be in MXFP4_G1_VARIANTS["fp4"];
    # 128 is only compiled without nt.
    return bm, bm != 128


# ---------------------------------------------------------------------------
# Pinned-path tuning: candidate space, per-shape CSV, tuner override
# ---------------------------------------------------------------------------
#: Per-(shape, token) winners for the pinned single-family path, produced by
#: ``op_tests/multigpu_tests/tune_mega_moe_TP.py``. The heuristic ladder above
#: is only the fallback for shapes this file does not cover: a fixed ladder
#: picked the best block_m in just 18 of 28 measured TP8 cells, and the misses
#: cost up to 2.55x, so the shapes that matter are tuned rather than guessed.
#:
#: ``inter_dim`` in this file is the **per-rank shard**, not the full tensor,
#: because that is what the pinned GEMMs actually see.
_PIN_TUNED_CSV = "flydsl_fuse_kernel_tuned_fmoe.csv"

#: Key columns; a row matches a call when all of these are equal.
_PIN_TUNED_KEYS = (
    "gfx",
    "cu_num",
    "tp",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
)

#: Set by the tuner to force one candidate, bypassing CSV and ladder alike.
_PIN_OVERRIDE: dict | None = None


def set_pin_override(choice: dict | None) -> None:
    """Force one pinned config, or restore CSV/ladder lookup with ``None``.

    Used by the tuner to walk the candidate space in-process; not a runtime
    knob. Callers must drop any cached plan afterwards (see
    :meth:`MegaMoeTPEngine.plan`, which bypasses its cache while an override is
    active).
    """
    global _PIN_OVERRIDE
    _PIN_OVERRIDE = choice


def pin_tuned_csv_path() -> str:
    """Absolute path of the pinned-path tuned config file.

    ``AITER_TP_MEGA_PIN_TUNED_CSV`` redirects it, so a tuning run can write a
    candidate file without overwriting the shipped one.
    """
    override = os.environ.get("AITER_TP_MEGA_PIN_TUNED_CSV")
    if override:
        return override
    import aiter.configs

    return os.path.join(os.path.dirname(aiter.configs.__file__), _PIN_TUNED_CSV)


@functools.lru_cache(maxsize=4)
def _load_pin_tuned(path: str, mtime: float) -> dict:
    """Read the pinned tuned CSV into ``{key tuple: choice dict}``.

    ``mtime`` is part of the cache key only so a tuner rewriting the file in the
    same process is picked up rather than served stale.
    """
    import csv as _csv

    table: dict = {}
    try:
        with open(path, newline="") as fh:
            for row in _csv.DictReader(fh):
                try:
                    key = _pin_tuned_key(
                        gfx=row["gfx"],
                        cu_num=int(row["cu_num"]),
                        tp=int(row["tp"]),
                        token=int(row["token"]),
                        model_dim=int(row["model_dim"]),
                        inter_dim=int(row["inter_dim"]),
                        expert=int(row["expert"]),
                        topk=int(row["topk"]),
                        act_type=row["act_type"],
                    )
                except (KeyError, ValueError):
                    continue  # malformed row: fall back rather than crash
                table[key] = {
                    "block_m": int(row["block_m"]),
                    "kernel1": row["kernelName1"],
                    "kernel2": row["kernelName2"],
                }
    except FileNotFoundError:
        pass
    return table


def _pin_tuned_key(**kw) -> tuple:
    return tuple(kw[name] for name in _PIN_TUNED_KEYS)


def pinned_candidates(cfg: "MegaMoeTPConfig") -> list[dict]:
    """Every legal pinned (GEMM1, GEMM2) tile choice for one shape.

    The search space the tuner walks. Constraints come from
    ``mxfp4_gemm1_kernels._validate`` (GEMM1) and ``stage2_rs_supported``
    (GEMM2), so every entry here is expected to compile and to keep the fused
    ReduceScatter available.
    """
    n_out = 2 * cfg.inter_dim
    out = []
    for bm, use_nt, inline in sorted(MXFP4_G1_VARIANTS["fp4"]):
        if inline:
            continue  # the FP4 wire hands GEMM1 an already-quantized operand
        for g1_bn in (64, 128, 256):
            # BN64 is compiled only for BM32 A4W4 non-inline.
            if g1_bn == 64 and bm != 32:
                continue
            if n_out % g1_bn:
                continue
            for g1_bk in (128, 256):
                if cfg.model_dim % g1_bk or cfg.model_dim // g1_bk > 32:
                    continue
                for g2_tn in (128, 256):
                    if cfg.model_dim % g2_tn:
                        continue
                    for g2_tk in (128, 256):
                        if cfg.inter_dim % g2_tk:
                            continue
                        for epilog in ("atomic", "reduce"):
                            for g2_nt in (True, False):
                                out.append(
                                    {
                                        "block_m": bm,
                                        "g1_nt": use_nt,
                                        "g1_bn": g1_bn,
                                        "g1_bk": g1_bk,
                                        "g2_tn": g2_tn,
                                        "g2_tk": g2_tk,
                                        "g2_epilog": epilog,
                                        "g2_nt": g2_nt,
                                    }
                                )
    return out


def pinned_default_choice(cfg: "MegaMoeTPConfig", bucket: int) -> dict:
    """The heuristic pinned config: widest even tile, ladder ``block_m``.

    What the pinned path runs for a shape the tuned CSV does not cover, and the
    baseline a tuning run measures against. Exposed so the tuner can time it
    explicitly rather than by clearing its override, which would otherwise read
    back the half-written CSV it is in the middle of producing.
    """
    bm, use_nt = _pinned_block_m(bucket)
    n_out = 2 * cfg.inter_dim
    # GEMM1 tiles: N over the gate/up axis, K over model_dim.
    g1_bn = 256 if n_out % 256 == 0 else 128
    g1_bk = 256 if cfg.model_dim % 256 == 0 else 128
    # GEMM2 tiles: N over model_dim, K over the inter shard.
    g2_tn = 256 if cfg.model_dim % 256 == 0 else 128
    g2_tk = 256 if cfg.inter_dim % 256 == 0 else 128
    if n_out % g1_bn or cfg.model_dim % g1_bk or cfg.inter_dim % g2_tk:
        raise ValueError(
            f"shape h{cfg.model_dim} i{cfg.inter_dim} does not tile onto the "
            "pinned GEMM1/GEMM2 families"
        )
    return {
        "block_m": bm,
        "g1_nt": use_nt,
        "g1_bn": g1_bn,
        "g1_bk": g1_bk,
        "g2_tn": g2_tn,
        "g2_tk": g2_tk,
        "g2_epilog": "atomic",
        "g2_nt": use_nt,
    }


def pinned_kernel_names(cfg: "MegaMoeTPConfig", choice: dict) -> tuple[str, str]:
    """(kernel1, kernel2) for one entry of :func:`pinned_candidates`."""
    from aiter.ops.flydsl.moe_kernels import build_flydslv2_gemm2_name

    bm = int(choice["block_m"])
    act = "_situv2" if cfg.activation == ActivationType.Situv2 else ""
    nt = "_nt" if choice["g1_nt"] else ""
    kernel1 = (
        f"flydsl_mxmoe_g1_a4w4_{bm}x{choice['g1_bn']}x{choice['g1_bk']}{nt}{act}"
    )
    kernel2 = build_flydslv2_gemm2_name(
        "fp4",
        "fp4",
        "bf16",
        tm=bm,
        epilog=choice["g2_epilog"],
        persist=False,
        use_nt=choice["g2_nt"],
        sbm=bm,
        tn=choice["g2_tn"],
        tk=choice["g2_tk"],
    )
    return kernel1, kernel2


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
#: Whether the ReduceScatter rides in GEMM2's tail (``auto``) or stays two
#: standalone launches (``off``).
RS_FUSE_MODES = ("auto", "off")


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
    #: ``auto`` folds the ReduceScatter into the GEMM2 kernel's tail whenever
    #: the tuned GEMM2 is a non-persistent atomic-epilogue row (the only kind
    #: that accumulates straight into the arena partial); ``off`` keeps the
    #: standalone publish + pull launches. ``AITER_TP_MEGA_RS_FUSE`` moves the
    #: default, so an A/B needs no code change.
    rs_fuse: str = field(
        default_factory=lambda: os.environ.get("AITER_TP_MEGA_RS_FUSE", "auto")
    )
    #: ``auto`` lets the AllGather push kernel do the MXFP4 quantization itself
    #: (one less launch, no staging buffer) whenever ``model_dim`` splits into
    #: whole 128-element per-thread units; ``off`` keeps the separate quant
    #: kernel. ``AITER_TP_MEGA_AGQ_FUSE`` moves the default.
    ag_quant_fuse: str = field(
        default_factory=lambda: os.environ.get("AITER_TP_MEGA_AGQ_FUSE", "auto")
    )

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
        if self.rs_fuse not in RS_FUSE_MODES:
            raise ValueError(f"rs_fuse must be one of {RS_FUSE_MODES}")
        if self.ag_quant_fuse not in RS_FUSE_MODES:
            raise ValueError(f"ag_quant_fuse must be one of {RS_FUSE_MODES}")

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
        if _PIN_OVERRIDE is not None:
            # The tuner walks candidates in-process; a cached plan would pin the
            # first one for the rest of the sweep.
            return self._resolve_plan(bucket)
        cached = self._plan_cache.get(bucket)
        if cached is not None:
            return cached
        plan = self._resolve_plan(bucket)
        self._plan_cache[bucket] = plan
        return plan


    # -- pinned single-family kernel selection --------------------------------
    def _pinned_row(self, bucket: int):
        """Build a (metadata, kernel1, kernel2) row on the two chosen families.

        GEMM1 is pinned to ``flydsl_mxmoe_g1_a4w4_*`` and GEMM2 to
        ``flydsl_moe2_layout_afp4_wfp4_*`` -- the only two families that expose a
        ``_composition`` hook, and therefore the only pair a single kernel can
        host. The stock tuned lookup is bypassed entirely: the point is one code
        path for every a4w4 shape.

        Staying on one family does not mean staying on one tile, though. Tiles
        come from :data:`_PIN_TUNED_CSV` where the shape is tuned, and from the
        widest-even-divisor heuristic otherwise.
        """
        from aiter.fused_moe import _make_mxfp4_metadata

        cfg = self.cfg
        # Precedence: tuner override > per-shape tuned CSV > heuristic ladder.
        if _PIN_OVERRIDE is not None:
            kernel1, kernel2 = pinned_kernel_names(cfg, _PIN_OVERRIDE)
            BM = int(_PIN_OVERRIDE["block_m"])
        else:
            tuned = self._pinned_tuned_lookup(bucket)
            if tuned is not None:
                kernel1, kernel2, BM = tuned
            else:
                kernel1, kernel2, BM = self._pinned_default(bucket)
        metadata = _make_mxfp4_metadata(
            kernel1, kernel2, GateMode.SEPARATED.value, 0, block_m=BM
        )
        return metadata, kernel1, kernel2

    def _pinned_tuned_lookup(self, bucket: int):
        """``(kernel1, kernel2, block_m)`` from the tuned CSV, or ``None``."""
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx

        cfg = self.cfg
        path = pin_tuned_csv_path()
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return None
        table = _load_pin_tuned(path, mtime)
        key = _pin_tuned_key(
            gfx=get_gfx(),
            cu_num=int(get_cu_num()),
            tp=int(cfg.world_size),
            token=int(bucket),
            model_dim=int(cfg.model_dim),
            inter_dim=int(cfg.inter_dim),
            expert=int(cfg.experts),
            topk=int(cfg.topk),
            act_type=str(cfg.activation),
        )
        row = table.get(key)
        if row is None:
            return None
        return row["kernel1"], row["kernel2"], int(row["block_m"])

    def _pinned_default(self, bucket: int):
        """Heuristic fallback: widest tile the shape divides, ladder block_m."""
        choice = pinned_default_choice(self.cfg, bucket)
        kernel1, kernel2 = pinned_kernel_names(self.cfg, choice)
        return kernel1, kernel2, int(choice["block_m"])

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

    def _fuses_rs(self, kernel2: str) -> bool:
        """Whether this GEMM2 can carry the ReduceScatter in its own tail."""
        if self.cfg.rs_fuse == "off":
            return False
        from aiter.ops.flydsl.kernels.mega_moe_tp.stage2_rs import stage2_rs_supported
        from aiter.ops.flydsl.mxfp4_kname import parse_flydsl_v2_gemm2_kernel

        try:
            return stage2_rs_supported(parse_flydsl_v2_gemm2_kernel(kernel2))
        except Exception:  # noqa: BLE001 - probe only, never fatal
            return False

    def _resolve_plan(self, bucket: int) -> "_CasePlan":
        cfg = self.cfg
        # An active tuner override implies the pinned path even when the env
        # switch is off, so a tuning run needs no extra environment setup.
        if _PIN_KERNELS or _PIN_OVERRIDE is not None:
            metadata, kernel1, kernel2 = self._pinned_row(bucket)
            return _CasePlan(
                bucket, "fp4_1x32", kernel1, kernel2, metadata, bucket,
                self._fuses_rs(kernel2),
            )
        _, kernel1, kernel2 = self._tuned_row(bucket)
        prequant_ok = _gemm1_takes_prequantized_fp4(kernel1)

        if cfg.ag_wire == "bf16":
            return _CasePlan(
                bucket,
                "bf16",
                kernel1,
                kernel2,
                None,
                bucket,
                self._fuses_rs(kernel2) if _BF16_VIA_IMPL else False,
            )
        if prequant_ok:
            # The FP4 wire always goes through ``_fused_moe_impl`` (it needs the
            # prequant metadata hook regardless), so the fused tail is free here.
            return _CasePlan(
                bucket, "fp4_1x32", kernel1, kernel2, None, bucket,
                self._fuses_rs(kernel2),
            )

        # The tuned row for this bucket quantizes inline, so it cannot read the
        # FP4 wire, and at BM16 no pre-quantized variant exists to retune onto.
        # Borrow a larger bucket's row instead -- both stages together, so the
        # GEMM1/GEMM2 scale-layout contract stays self-consistent.
        # Borrowing is a measured loss wherever it fires, so `auto` -- which is
        # supposed to pick the faster path -- declines it. kimi3 TP8, e2e us:
        #
        #   tokens    bf16 wire (own tuned row)    fp4 wire (borrowed row)
        #        8                        246.1                      304.1
        #       64                        352.9                      355.9
        #
        # The borrowed GEMM row costs +76 us at M=8 while the FP4 wire saves
        # 0.2 us of AllGather, because at that size the collective is
        # latency-bound and a 3.77x smaller wire row buys nothing. At the sizes
        # where the AllGather *is* bandwidth-bound (512, 4096) no borrow is
        # needed and the FP4 wire wins by 6 and 123 us -- which is why this
        # declines the borrow rather than the wire. `ag_wire='fp4_1x32'` still
        # forces it for anyone who wants the smaller wire regardless.
        borrowed = None if cfg.ag_wire == "auto" else self._probe_prequant_row(bucket)
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
            return _CasePlan(
                bucket,
                "fp4_1x32",
                sub1,
                sub2,
                metadata,
                src,
                self._fuses_rs(sub2),
            )

        if cfg.ag_wire == "fp4_1x32":
            raise ValueError(
                f"ag_wire='fp4_1x32' needs a pre-quantized GEMM1, but M={bucket} "
                f"resolves to {kernel1!r}, which quantizes inline, and no larger "
                "token bucket for this shape resolves to one either. Use "
                "ag_wire='bf16' or tune a non-f16in GEMM1 for this shape."
            )
        return _CasePlan(
            bucket,
            "bf16",
            kernel1,
            kernel2,
            None,
            bucket,
            self._fuses_rs(kernel2) if _BF16_VIA_IMPL else False,
        )

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
            #
            # Better still, it is row-local arithmetic with no cross-thread
            # dependency, so where the shape allows it the push kernel does the
            # quantization itself: one less launch and one less round trip
            # through the staging buffer.
            if cfg.ag_quant_fuse != "off" and comm.quant_push_available():
                gathered = comm.all_gather_quant(x_local, topk_ids, topk_weights)
                return (
                    gathered.payload.view(AQ_DTYPE),
                    gathered.scale.view(dtypes.fp8_e8m0),
                    gathered.topk_weights,
                    gathered.topk_ids,
                )
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

    def _fused_stage2(self, plan: "_CasePlan", local_tokens: int):
        """A ``metadata.stage2`` that also runs this rank's ReduceScatter.

        GEMM2's atomic epilogue already accumulates into the arena partial, so
        the only thing standing between it and the ReduceScatter is "have all
        peers landed". Folding that into the same kernel turns two launches
        into a device-side counter. Returns ``(stage2, result_box)``; after the
        MoE call ``result_box[0]`` holds this rank's output shard.
        """
        from aiter.ops.flydsl.kernels.mega_moe_tp.stage2_rs import run_stage2_rs
        from aiter.ops.flydsl.mxfp4_kname import parse_flydsl_v2_gemm2_kernel

        cfg = self.cfg
        comm = self._collectives(plan.ag_wire)
        kernel_cfg = parse_flydsl_v2_gemm2_kernel(plan.gemm2_kernel)
        box: list = [None]
        if kernel_cfg["epilog"] == "reduce":
            return self._fused_stage2_reduce(plan, local_tokens, comm, kernel_cfg, box)

        def stage2(
            inter_states,
            w1,
            w2,
            sorted_token_ids,
            sorted_expert_ids,
            num_valid_ids,
            moe_out,
            topk,
            *,
            w2_scale=None,
            a2_scale=None,
            block_m=None,
            sorted_weights=None,
            kernelName2="",
            **_kwargs,
        ):
            box[0] = run_stage2_rs(
                inter_states=inter_states,
                a2_scale=a2_scale,
                w2=w2,
                w2_scale=w2_scale,
                sorted_expert_ids=sorted_expert_ids,
                num_valid_ids=num_valid_ids,
                sorted_token_ids=sorted_token_ids,
                sorted_weights=sorted_weights,
                partial=moe_out,
                output=comm.output_buffer(local_tokens),
                desc_ptr=comm.rs_descriptor(),
                rank=cfg.rank,
                tp_size=cfg.world_size,
                local_rows=local_tokens,
                M_logical=moe_out.shape[0],
                NE=w2.shape[0],
                model_dim=moe_out.shape[1],
                inter_dim=w1.shape[1] // 2 if w1 is not None else cfg.inter_dim,
                topk=topk,
                kernel_cfg=kernel_cfg,
                block_m=block_m,
            )
            return moe_out

        return functools.partial(stage2, kernelName2=plan.gemm2_kernel), box


    def _fused_stage2_reduce(self, plan, local_tokens, comm, kernel_cfg, box):
        """The reduce-epilogue twin of :meth:`_fused_stage2`.

        A ``reduce`` GEMM2 stages per-route rows and a reduction kernel turns
        them into the ``[M, H]`` partial, so the ReduceScatter rides in *that*
        kernel instead. Everything before it mirrors
        ``_flydsl_v2_stage2_wrapper``'s reduce branch, so the GEMM and the
        intermediate format are the tuned ones.
        """
        import os as _os

        import torch as _torch

        from aiter.fused_moe import _flydsl_stage2_fp8_enabled, _mxfp4_scale_u8
        from aiter.ops.flydsl.kernels.mega_moe_tp.reduce_rs import run_reduce_rs
        from aiter.ops.flydsl.kernels.mxfp4_gemm_common import (
            FP8OUT_PITCH_ALIGN,
            fp8out_row_bytes,
            fp8out_scale_blk,
        )
        from aiter.ops.flydsl.kernels.mxmoe_dispatcher import mxfp4_moe_gemm2

        cfg = self.cfg

        def stage2(
            inter_states,
            w1,
            w2,
            sorted_token_ids,
            sorted_expert_ids,
            num_valid_ids,
            moe_out,
            topk,
            *,
            w2_scale=None,
            a2_scale=None,
            block_m=None,
            sorted_weights=None,
            topk_weights=None,
            kernelName2="",
            **_kwargs,
        ):
            token_num = moe_out.shape[0]
            model_dim = moe_out.shape[1]
            inter_dim = w1.shape[1] // 2 if w1 is not None else cfg.inter_dim
            kstatic = _os.environ.get("MXFP4_G2_KSTATIC", "1") == "1"
            fp8_inter = _flydsl_stage2_fp8_enabled()
            if fp8_inter and kstatic:
                fp8_inter = sorted_weights is not None and topk_weights is not None
            defer_weight = fp8_inter and kstatic
            scale_blk = pitch_align = None
            if fp8_inter:
                scale_blk = fp8out_scale_blk(model_dim) if kstatic else 8
                pitch_align = FP8OUT_PITCH_ALIGN if kstatic else 0
                target = _torch.empty(
                    (
                        token_num * topk,
                        fp8out_row_bytes(
                            model_dim, scale_blk=scale_blk, pitch_align=pitch_align
                        ),
                    ),
                    dtype=_torch.uint8,
                    device=moe_out.device,
                )
            else:
                target = _torch.empty(
                    (token_num, topk, model_dim),
                    dtype=moe_out.dtype,
                    device=moe_out.device,
                )
            mxfp4_moe_gemm2(
                inter_sorted_quant=_mxfp4_scale_u8(inter_states),
                inter_sorted_shuffled_scale=_mxfp4_scale_u8(a2_scale),
                w2_u8=_mxfp4_scale_u8(w2),
                w2_scale_u8=_mxfp4_scale_u8(w2_scale),
                sorted_expert_ids=sorted_expert_ids,
                cumsum_tensor=num_valid_ids,
                sorted_token_ids=sorted_token_ids,
                sorted_weights=sorted_weights,
                out=target,
                M_logical=token_num,
                max_sorted=inter_states.shape[0],
                NE=w2.shape[0],
                D_HIDDEN=model_dim,
                D_INTER=inter_dim,
                topk=topk,
                BM=kernel_cfg["tile_m"],
                BN=kernel_cfg["tile_n"],
                BK=kernel_cfg["tile_k"],
                use_nt=kernel_cfg["use_nt"],
                a_dtype=kernel_cfg["a_dtype"],
                b_dtype=kernel_cfg["b_dtype"],
                epilog="reduce",
                SBM=kernel_cfg["sort_block_m"]
                or (int(block_m) if block_m else kernel_cfg["tile_m"]),
                persist=kernel_cfg["persist"],
                g2_bf16_lds=kernel_cfg["bf16_lds"],
                g2_spart=kernel_cfg["spart"],
                out_dtype="fp8" if fp8_inter else "bf16",
            )
            box[0] = run_reduce_rs(
                target=target,
                partial=moe_out,
                output=comm.output_buffer(local_tokens),
                token_num=token_num,
                topk=topk,
                model_dim=model_dim,
                tp_size=cfg.world_size,
                rank=cfg.rank,
                local_rows=local_tokens,
                desc_ptr=comm.rs_descriptor(),
                is_fp8=fp8_inter,
                topk_weights=topk_weights if defer_weight else None,
                fp8_scale_blk=scale_blk,
                fp8_pitch_align=pitch_align,
            )
            return moe_out

        stage2._is_flydsl_v2_stage2 = True
        return functools.partial(stage2, kernelName2=plan.gemm2_kernel), box


    def _fuses_stage12(self, plan) -> bool:
        """Whether this bucket's tuned pair can share one kernel."""
        from aiter.ops.flydsl.kernels.mega_moe_tp.stage12_rs import stage12_supported
        from aiter.ops.flydsl.mxfp4_kname import parse_flydsl_v2_gemm2_kernel

        try:
            if not _is_mxfp4_kname(plan.gemm1_kernel):
                return False
            return stage12_supported(
                _parse_mxfp4_g1_kname(plan.gemm1_kernel),
                parse_flydsl_v2_gemm2_kernel(plan.gemm2_kernel),
                self.cfg.model_dim,
            )
        except Exception:  # noqa: BLE001 - probe only, never fatal
            return False

    def _fused_stage12(self, plan, local_tokens, comm, g2_cfg):
        """Replace stage1+stage2 with one GEMM1+GEMM2+ReduceScatter kernel.

        ``stage1`` keeps its own identity and keywords -- ``fused_moe_2stages``
        dispatches on ``metadata.stage1.func``, and the operand derivation there
        is what this path reuses -- but its ``_gemm1_launch`` hook diverts the
        launch into a dict. ``stage2`` then runs both GEMMs and the tail in one
        kernel from that dict. Returns ``(transform, box)``.
        """
        from aiter.ops.flydsl.kernels.mega_moe_tp.stage12_rs import run_stage12_rs

        cfg = self.cfg
        box: list = [None]
        captured: dict = {}

        def capture_gemm1(**kwargs):
            captured.update(kwargs)

        def stage2(
            inter_states,
            w1,
            w2,
            sorted_token_ids,
            sorted_expert_ids,
            num_valid_ids,
            moe_out,
            topk,
            *,
            w2_scale=None,
            a2_scale=None,
            block_m=None,
            sorted_weights=None,
            kernelName2="",
            **_kwargs,
        ):
            box[0] = run_stage12_rs(
                g1=captured,
                w2=w2,
                w2_scale=w2_scale,
                sorted_token_ids=sorted_token_ids,
                sorted_weights=sorted_weights,
                partial=moe_out,
                output=comm.output_buffer(local_tokens),
                desc_ptr=comm.rs_descriptor(),
                rank=cfg.rank,
                tp_size=cfg.world_size,
                local_rows=local_tokens,
                M_logical=moe_out.shape[0],
                model_dim=moe_out.shape[1],
                inter_dim=w1.shape[1] // 2 if w1 is not None else cfg.inter_dim,
                g2_cfg=g2_cfg,
                block_m=block_m,
            )
            return moe_out

        stage2._is_flydsl_v2_stage2 = True
        stage2_partial = functools.partial(stage2, kernelName2=plan.gemm2_kernel)

        def transform(metadata):
            stage1 = functools.partial(
                metadata.stage1.func,
                **metadata.stage1.keywords,
                _gemm1_launch=capture_gemm1,
            )
            return replace(metadata, stage1=stage1, stage2=stage2_partial)

        return transform, box

    def local_moe(
        self, a1, a1_scale, topk_weights_all, topk_ids_all, plan=None, local_tokens=0
    ):
        """Run GEMM1 + activation + GEMM2 + weighted top-k reduce for all tokens.

        The result lands directly in the symmetric arena, so the ReduceScatter
        reads it in place instead of staging a copy.  GEMM2's atomic epilogue
        needs a zeroed target and ``moe_sorting`` zeroes whatever output buffer
        it is handed, so the arena slice is safe to reuse every call.

        With ``plan.fuse_rs`` the ReduceScatter rides in GEMM2's own tail and
        this returns ``(partial, y_local)``; otherwise it returns
        ``(partial, None)`` and the caller runs the standalone collective.

        The BF16 wire hands the activation to the ordinary public entry point;
        only the MXFP4 wire needs the private one, to force the pre-quantized
        activation path (``fused_moe`` exposes no hook for that).
        """
        total = int(topk_ids_all.shape[0])
        plan = plan or self.plan(total)
        output = self._collectives(plan.ag_wire).partial_buffer(total)
        # The BF16 wire takes the *public* entry point. ``_fused_moe_impl`` is
        # the only one that accepts ``_metadata_transform``, and so the only way
        # to reach the fused ReduceScatter -- but it is a net loss in eager
        # mode. Measured both ways (e2e us): kimi3 64 tokens 352.9 -> 357.3, and
        # glm5 64 tokens 234.6 -> **290.2**. The cost is host-side and shape
        # dependent, not a property of the tail: under CUDA-graph capture the
        # two entry points are identical (95.8 vs 96.0 us at 8 tokens) and the
        # fused tail comes for free. So this is the right default for eager
        # execution and the wrong one under graphs -- revisit if this layer is
        # deployed captured. ``AITER_TP_MEGA_BF16_VIA_IMPL=1`` flips it.
        # The merged GEMM1+GEMM2+RS kernel only exists for the BM16 inline-quant
        # pair, which lives on this wire -- so taking it is what makes the wire
        # worth routing through the hookable entry point despite its host cost.
        fuse12 = _STAGE12 and a1_scale is None and self._fuses_stage12(plan)
        if a1_scale is None and not (_BF16_VIA_IMPL or fuse12):
            return (
                fused_moe(
                    a1,
                    self.w1,
                    self.w2,
                    topk_weights_all,
                    topk_ids_all,
                    output=output,
                    **self._moe_kwargs,
                ),
                None,
            )

        from aiter.fused_moe import _fused_moe_impl

        kwargs = dict(self._moe_kwargs)
        kwargs["quant_type"] = kwargs["quant_type"].value
        kwargs["activation"] = kwargs["activation"].value
        # The activation arrives packed FP4, so the output dtype cannot be
        # inferred from it the way the BF16 wire allows.
        kwargs["dtype"] = dtypes.bf16
        if a1_scale is None:
            transform = None
        elif plan.metadata is None:
            transform = _prequant_transform
        else:
            # This bucket's own tuned row quantizes inline; run the borrowed
            # row instead of whatever the lookup inside _fused_moe_impl finds.
            borrowed = replace(plan.metadata, prequant=True)
            transform = lambda _metadata: borrowed  # noqa: E731
        if fuse12 and local_tokens > 1:
            from aiter.ops.flydsl.mxfp4_kname import parse_flydsl_v2_gemm2_kernel

            stage12_transform, box = self._fused_stage12(
                plan,
                local_tokens,
                self._collectives(plan.ag_wire),
                parse_flydsl_v2_gemm2_kernel(plan.gemm2_kernel),
            )
            partial = _fused_moe_impl(
                a1,
                self.w1,
                self.w2,
                topk_weights_all,
                topk_ids_all,
                a1_scale=a1_scale,
                output=output,
                _q_dtype_a=AQ_DTYPE if a1_scale is not None else None,
                _metadata_transform=stage12_transform,
                **kwargs,
            )
            return partial, box[0]

        box = None
        # Two exclusions, both measured:
        #
        # * ``local_tokens == 1`` -- the fused tail's pull returns garbage there
        #   (rel_l2 1.0) while leaving the partial itself intact; bisected to
        #   the pull, with 8/16/32 local rows all correct. A decode-with-TP8
        #   corner, gated rather than chased.
        # * past the ReduceScatter byte budget -- the fused pull is slower than
        #   NCCL there, and a tail has no fallback of its own.
        fuse_tail = (
            plan.fuse_rs
            and local_tokens > 1
            and self._collectives(plan.ag_wire).rs_fused_is_profitable(local_tokens)
        )
        if fuse_tail:
            stage2, box = self._fused_stage2(plan, local_tokens)
            base = transform
            transform = (
                (lambda md: replace(md, stage2=stage2))
                if base is None
                else (lambda md: replace(base(md), stage2=stage2))
            )
        if transform is None:
            transform = lambda md: md  # noqa: E731
        partial = _fused_moe_impl(
            a1,
            self.w1,
            self.w2,
            topk_weights_all,
            topk_ids_all,
            a1_scale=a1_scale,
            output=output,
            _q_dtype_a=AQ_DTYPE if a1_scale is not None else None,
            _metadata_transform=transform,
            **kwargs,
        )
        return partial, (box[0] if box is not None else None)  # None -> caller RS

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
        _, y_local = self.local_moe(a1, a1_scale, wts, ids, plan=plan, local_tokens=m)
        if y_local is not None:
            return y_local
        return self.reduce_scatter(m, plan)

    __call__ = forward


@dataclass(frozen=True)
class _CasePlan:
    """The per-token-bucket decisions the runtime caches.

    ``metadata`` is the two-stage config to force, set only when this bucket's
    own tuned row had to be swapped out to stay on the FP4 wire; ``None`` means
    the ordinary lookup already lands on the right row.  ``gemm_bucket`` records
    which bucket the kernels came from, so it is visible in tests and logs when
    it is not ``tokens``.  ``fuse_rs`` records whether this bucket's GEMM2 runs
    the ReduceScatter in its own tail instead of as two extra launches.
    """

    tokens: int
    ag_wire: str
    gemm1_kernel: str
    gemm2_kernel: str
    metadata: object | None = None
    gemm_bucket: int = 0
    fuse_rs: bool = False


def _prequant_transform(metadata):
    """Force the pre-quantized activation path for an FP4 wire.

    ``_make_mxfp4_metadata`` only sets ``prequant`` for FP8 activations, and the
    caller-side promotion in ``fused_moe_2stages`` skips ``block_m == 16``
    because that is the inline-quant variant.  Here the wire format has already
    been validated against the GEMM1 variant, so the promotion is unconditional.
    """
    return replace(metadata, prequant=True)
