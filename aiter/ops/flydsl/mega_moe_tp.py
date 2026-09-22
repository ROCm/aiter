# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused tensor-parallel MoE layer: AllGather + GEMM1 + GEMM2 + ReduceScatter."""

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

_STAGE2_TARGETS = {}


def _stage2_target(shape, dtype, device):
    """Staging buffer for the ``reduce`` epilogue's GEMM2 output."""
    key = (tuple(int(x) for x in shape), str(dtype), str(device))
    buf = _STAGE2_TARGETS.get(key)
    if buf is None:
        buf = torch.empty(tuple(int(x) for x in shape), dtype=dtype, device=device)
        _STAGE2_TARGETS[key] = buf
    return buf


_TIME_STEPS = os.environ.get("AITER_TP_MEGA_TIME", "0") == "1"
_TIME_AG = os.environ.get("AITER_TP_MEGA_TIME_AG", "0") == "1"
_STAGE12 = os.environ.get("AITER_TP_MEGA_STAGE12", "0") == "1"
_STAGE12_MIN_M = int(os.environ.get("AITER_TP_MEGA_STAGE12_MIN_M", "1"))
_MEGA_AG_ENV = os.environ.get("AITER_TP_MEGA_FUSE_AG", "0")
_MEGA_AG = _MEGA_AG_ENV in ("1", "force")
_MEGA_FORCE = _MEGA_AG_ENV == "force"
_PIN_KERNELS = os.environ.get("AITER_TP_MEGA_PIN_KERNELS", "0") == "1"
_WAVES_PER_EU_DEFAULT = int(os.environ.get("AITER_TP_MEGA_WAVES_PER_EU", "2"))
_PIN_BM = os.environ.get("AITER_TP_MEGA_PIN_BM", "auto")
_PIN_BM_LADDER = ((1024, 32), (2048, 64), (1 << 30, 128))


def _pinned_block_m(bucket: int) -> tuple[int, bool]:
    """(block_m, use_nt) for one token bucket on the pinned GEMM1 family."""
    if _PIN_BM != "auto":
        bm = int(_PIN_BM)
    else:
        bm = next(b for limit, b in _PIN_BM_LADDER if bucket <= limit)
    if bm not in (16, 32, 64, 128):
        raise ValueError(
            f"pinned block_m must be 16/32/64/128 (fp4 prequant), got {bm}"
        )
    return bm, bm != 128


_PIN_TUNED_CSV = "flydsl_fuse_kernel_tuned_fmoe.csv"

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

_PIN_OVERRIDE: dict | None = None


_MEGA_OVERRIDE: bool | None = None
_STAGE12_OVERRIDE: bool | None = None


def _stage12_on() -> bool:
    return _STAGE12 if _STAGE12_OVERRIDE is None else _STAGE12_OVERRIDE


def set_mega_override(value: bool | None) -> None:
    """Force the merged kernel on or off, or restore the per-shape lookup."""
    global _MEGA_OVERRIDE
    _MEGA_OVERRIDE = value


def set_stage12_override(value: bool | None) -> None:
    """Force the merged GEMM1+GEMM2+RS kernel on or off, or restore the env."""
    global _STAGE12_OVERRIDE
    _STAGE12_OVERRIDE = value


def set_pin_override(choice: dict | None) -> None:
    """Force one pinned config, or restore CSV/ladder lookup with ``None``."""
    global _PIN_OVERRIDE
    _PIN_OVERRIDE = choice


def pin_tuned_csv_path() -> str:
    """Absolute path of the pinned-path tuned config file."""
    override = os.environ.get("AITER_TP_MEGA_PIN_TUNED_CSV")
    if override:
        return override
    import aiter.configs

    return os.path.join(os.path.dirname(aiter.configs.__file__), _PIN_TUNED_CSV)


@functools.lru_cache(maxsize=4)
def _load_pin_tuned(path: str, mtime: float) -> dict:
    """Read the pinned tuned CSV into ``{key tuple: choice dict}``."""
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
                    continue
                table[key] = {
                    "block_m": int(row["block_m"]),
                    "kernel1": row["kernelName1"],
                    "kernel2": row["kernelName2"],
                    "mega": str(row.get("mega", "1")).strip() not in ("0", "false"),
                    "waves_per_eu": _int_or(row.get("waves_per_eu"), 0),
                    "stage12": str(row.get("stage12", "1")).strip()
                    not in ("0", "false"),
                }
    except FileNotFoundError:
        pass
    return table


def _int_or(value, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _pin_tuned_key(**kw) -> tuple:
    return tuple(kw[name] for name in _PIN_TUNED_KEYS)


def pinned_candidates(cfg: "MegaMoeTPConfig") -> list[dict]:
    """Every legal pinned (GEMM1, GEMM2) tile choice for one shape."""
    n_out = 2 * cfg.inter_dim
    out = []
    for bm, use_nt, inline in sorted(MXFP4_G1_VARIANTS["fp4"]):
        if inline:
            continue
        for g1_bn in (64, 128, 256):
            if g1_bn == 64 and bm != 32:
                continue
            if n_out % g1_bn:
                continue
            for g1_bk in (256,):
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
                                for wpe in (0, 2):
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
                                            "waves_per_eu": wpe,
                                        }
                                    )
    return out


def pinned_default_choice(cfg: "MegaMoeTPConfig", bucket: int) -> dict:
    """The heuristic pinned config: widest even tile, ladder ``block_m``."""
    bm, use_nt = _pinned_block_m(bucket)
    n_out = 2 * cfg.inter_dim
    g1_bn = 256 if n_out % 256 == 0 else 128
    g1_bk = 256 if cfg.model_dim % 256 == 0 else 128
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
        "waves_per_eu": _WAVES_PER_EU_DEFAULT,
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

AG_WIRE_MODES = ("auto", "fp4_1x32", "bf16")
RS_WIRE_MODES = ("auto", "bf16")
RS_FUSE_MODES = ("auto", "off")


@dataclass(frozen=True)
class MegaMoeTPConfig:
    """Everything that is fixed for the lifetime of one fused TP MoE layer."""

    rank: int
    world_size: int
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    max_local_tokens: int
    activation: ActivationType = ActivationType.Situv2
    beta: float | None = None
    linear_beta: float | None = None
    ag_wire: str = "auto"
    rs_wire: str = "auto"
    rs_fuse: str = field(
        default_factory=lambda: os.environ.get("AITER_TP_MEGA_RS_FUSE", "auto")
    )
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


def _gemm1_takes_prequantized_fp4(kernel_name1: str) -> bool:
    """Can this GEMM1 consume an already-MXFP4-quantized A operand?"""
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
    """Dig the first matching keyword out of a (possibly nested) functools.partial."""
    seen = 0
    while fn is not None and seen < 8:
        kwargs = getattr(fn, "keywords", None) or {}
        for key in keys:
            if kwargs.get(key):
                return str(kwargs[key])
        fn = getattr(fn, "func", None)
        seen += 1
    return ""


class MegaMoeTP:
    """Fused AllGather + GEMM1 + GEMM2 + ReduceScatter MoE for one TP rank."""

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

        self._comm: dict[str, TpMoeCollectives] = {}

        self._quant = get_hip_quant(QUANT_TYPE)
        self._sort_bufs: dict = {}
        situ = cfg.activation == ActivationType.Situv2
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

    def plan(self, global_tokens: int) -> "_CasePlan":
        """Resolve (and cache) the wire format and kernel names for one M."""
        from aiter.fused_moe import get_padded_M

        bucket = int(get_padded_M(global_tokens))
        if _PIN_OVERRIDE is not None:
            return self._resolve_plan(bucket)
        cached = self._plan_cache.get(bucket)
        if cached is not None:
            return cached
        plan = self._resolve_plan(bucket)
        self._plan_cache[bucket] = plan
        return plan


    def _pinned_row(self, bucket: int):
        """Build a (metadata, kernel1, kernel2, waves_per_eu) row."""
        from aiter.fused_moe import _make_mxfp4_metadata

        cfg = self.cfg
        if _PIN_OVERRIDE is not None:
            kernel1, kernel2 = pinned_kernel_names(cfg, _PIN_OVERRIDE)
            BM = int(_PIN_OVERRIDE["block_m"])
            wpe = int(_PIN_OVERRIDE.get("waves_per_eu", _WAVES_PER_EU_DEFAULT))
            s12 = True
        else:
            tuned = self._pinned_tuned_lookup(bucket)
            if tuned is not None:
                kernel1, kernel2, BM, wpe, s12 = tuned
            else:
                kernel1, kernel2, BM = self._pinned_default(bucket)
                wpe = _WAVES_PER_EU_DEFAULT
                s12 = True
        metadata = _make_mxfp4_metadata(
            kernel1, kernel2, GateMode.SEPARATED.value, 0, block_m=BM
        )
        return metadata, kernel1, kernel2, wpe, s12

    def _pinned_tuned_lookup(self, bucket: int):
        """``(kernel1, kernel2, block_m, waves_per_eu)`` from the CSV, or None."""
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
        return (
            row["kernel1"],
            row["kernel2"],
            int(row["block_m"]),
            int(row["waves_per_eu"]),
            bool(row["stage12"]),
        )

    def _mega_allowed_by_csv(self, bucket: int) -> bool:
        """Whether the tuned row opts this shape into the merged kernel."""
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx

        cfg = self.cfg
        path = pin_tuned_csv_path()
        try:
            table = _load_pin_tuned(path, os.path.getmtime(path))
        except OSError:
            return True
        row = table.get(
            _pin_tuned_key(
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
        )
        return True if row is None else bool(row["mega"])

    def _pinned_default(self, bucket: int):
        """Heuristic fallback: widest tile the shape divides, ladder block_m."""
        choice = pinned_default_choice(self.cfg, bucket)
        kernel1, kernel2 = pinned_kernel_names(self.cfg, choice)
        return kernel1, kernel2, int(choice["block_m"])

    def _tuned_row(self, bucket: int):
        """Look up the tuned two-stage config for one token bucket."""
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
                True,
                cfg.activation,
                False,
                0,
                0,
                True,
                GateMode.SEPARATED.value,
            )
        except Exception as exc:
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
        """First bucket above ``bucket`` whose tuned GEMM1 reads pre-quantized A."""
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
        except Exception:
            return False

    def _mega_ag(self, plan) -> bool:
        """Whether the merged kernel also hosts the quantize-and-AllGather."""
        if not _stage12_on():
            return False
        if not (_MEGA_AG or _MEGA_OVERRIDE):
            return False
        if plan.ag_wire != "fp4_1x32":
            return False
        if not self._fuses_stage12(plan):
            return False
        if _MEGA_OVERRIDE is not None:
            if not _MEGA_OVERRIDE:
                return False
        elif not (_MEGA_FORCE or self._mega_allowed_by_csv(plan.tokens)):
            return False
        from aiter.ops.flydsl.kernels.mega_moe_tp.allgather_quant_push import (
            quant_push_supported,
        )

        return quant_push_supported(self.cfg.model_dim, self.cfg.topk)

    def stage12_mode(self, plan, local_tokens: int) -> str:
        """Why the merged GEMM1+GEMM2+RS kernel does or does not run."""
        if not _stage12_on():
            return "off"
        if plan.ag_wire != "fp4_1x32":
            return "bf16-wire"
        if not self._fuses_stage12(plan):
            return "unsupported"
        if local_tokens < _STAGE12_MIN_M:
            return f"m{local_tokens}-skip"
        return "mega" if self._mega_ag(plan) else "fused"

    def _resolve_plan(self, bucket: int) -> "_CasePlan":
        cfg = self.cfg
        if _PIN_KERNELS or _PIN_OVERRIDE is not None:
            metadata, kernel1, kernel2, wpe, s12 = self._pinned_row(bucket)
            return _CasePlan(
                bucket, "fp4_1x32", kernel1, kernel2, metadata, bucket,
                self._fuses_rs(kernel2), wpe, s12,
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
                False,
            )
        if prequant_ok:
            return _CasePlan(
                bucket, "fp4_1x32", kernel1, kernel2, None, bucket,
                self._fuses_rs(kernel2),
            )

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
            False,
        )

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
        """Gather the activation shard (and the route) into the global token set."""
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
            if self._mega_ag(plan):
                if _TIME_AG and not torch.cuda.is_current_stream_capturing():
                    sub = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                    sub[0].record()
                    weights_all, ids_all = comm.all_gather_route(
                        topk_ids, topk_weights
                    )
                    sub[1].record()
                    comm.publish_payload_source(x_local, topk_ids, topk_weights)
                    sub[2].record()
                    torch.cuda.synchronize()
                    self._ag_sub_us = (
                        sub[0].elapsed_time(sub[1]) * 1e3,
                        sub[1].elapsed_time(sub[2]) * 1e3,
                    )
                    payload, scale = comm.payload_views(total)
                    return (
                        payload.view(AQ_DTYPE),
                        scale.view(dtypes.fp8_e8m0),
                        weights_all,
                        ids_all,
                    )
                weights_all, ids_all = comm.all_gather_route(
                    topk_ids, topk_weights
                )
                comm.publish_payload_source(x_local, topk_ids, topk_weights)
                payload, scale = comm.payload_views(total)
                return (
                    payload.view(AQ_DTYPE),
                    scale.view(dtypes.fp8_e8m0),
                    weights_all,
                    ids_all,
                )
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
        """A ``metadata.stage2`` that also runs this rank's ReduceScatter."""
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
        """The reduce-epilogue twin of :meth:`_fused_stage2`."""
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
                target = _stage2_target(
                    (
                        token_num * topk,
                        fp8out_row_bytes(
                            model_dim, scale_blk=scale_blk, pitch_align=pitch_align
                        ),
                    ),
                    _torch.uint8,
                    moe_out.device,
                )
            else:
                target = _stage2_target(
                    (token_num, topk, model_dim), moe_out.dtype, moe_out.device
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
        """Whether this bucket's tuned pair can *and should* share one kernel."""
        if not plan.stage12:
            return False
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
        except Exception:
            return False

    def _alloc_sorting(
        self,
        topk_ids,
        topk_weight,
        num_experts,
        model_dim,
        dtype,
        block_size,
        *,
        accumulate=True,
        output_aux=False,
        output=None,
        **_kwargs,
    ):
        """A ``moe_sorting`` stand-in for a kernel that sorts itself."""
        topk = int(topk_ids.shape[1])
        device = topk_ids.device
        padded = int(topk_ids.numel() + num_experts * block_size - topk)
        blocks = int((padded + block_size - 1) // block_size)
        key = (padded, blocks, int(topk_ids.numel()))
        buf = self._sort_bufs.get(key)
        if buf is None:
            i32 = dict(dtype=dtypes.i32, device=device)
            buf = (
                torch.empty(padded, **i32),
                torch.empty(padded, dtype=dtypes.fp32, device=device),
                torch.empty(blocks, **i32),
                torch.zeros(2, **i32),
                torch.empty(padded, **i32),
                torch.empty(int(topk_ids.numel()), **i32),
            )
            self._sort_bufs[key] = buf
        sorted_ids, sorted_weights, sorted_eids, num_valid, m_indices, rev = buf
        moe_buf = (
            output
            if output is not None
            else torch.empty(
                (int(topk_ids.shape[0]), int(model_dim)), dtype=dtype, device=device
            )
        )
        if accumulate:
            moe_buf.zero_()
        ret = (sorted_ids, sorted_weights, sorted_eids, num_valid, moe_buf)
        if output_aux:
            return (*ret, m_indices, rev)
        return ret

    def _fused_stage12(
        self,
        plan,
        local_tokens,
        comm,
        g2_cfg,
        base_transform=None,
        tk_ids=None,
        tk_weights=None,
    ):
        """Replace stage1+stage2 with one GEMM1+GEMM2+ReduceScatter kernel."""
        from aiter.ops.flydsl.kernels.mega_moe_tp.stage12_rs import (
            hosts_reduce_as_atomic,
            run_stage12_rs,
        )

        cfg = self.cfg
        box: list = [None]
        captured: dict = {}
        mega_ag = self._mega_ag(plan)
        zero_partial = g2_cfg["epilog"] == "reduce" and hosts_reduce_as_atomic()
        raw_scale = comm.payload_views(local_tokens * cfg.world_size)[1] if mega_ag else None

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
            if zero_partial:
                moe_out.zero_()
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
                ag_desc_ptr=comm.ag_descriptor() if mega_ag else 0,
                ascale_raw=raw_scale if mega_ag else None,
                topk=cfg.topk,
                fuse_ag=mega_ag,
                waves_per_eu=plan.waves_per_eu,
                tk_ids=tk_ids,
                tk_weights=tk_weights,
                num_valid=num_valid_ids,
            )
            return moe_out

        stage2._is_flydsl_v2_stage2 = True
        stage2_partial = functools.partial(stage2, kernelName2=plan.gemm2_kernel)

        def transform(metadata):
            base = metadata if base_transform is None else base_transform(metadata)
            stage1 = functools.partial(
                base.stage1.func,
                **base.stage1.keywords,
                _gemm1_launch=capture_gemm1,
            )
            return replace(base, stage1=stage1, stage2=stage2_partial)

        return transform, box

    def local_moe(
        self, a1, a1_scale, topk_weights_all, topk_ids_all, plan=None, local_tokens=0
    ):
        """Run GEMM1 + activation + GEMM2 + weighted top-k reduce for all tokens."""
        total = int(topk_ids_all.shape[0])
        plan = plan or self.plan(total)
        output = self._collectives(plan.ag_wire).partial_buffer(total)
        fuse12 = _stage12_on() and a1_scale is not None and self._fuses_stage12(plan)
        if a1_scale is None and not fuse12:
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
        kwargs["dtype"] = dtypes.bf16
        if a1_scale is None:
            transform = None
        elif plan.metadata is None:
            transform = _prequant_transform
        else:
            borrowed = replace(plan.metadata, prequant=True)
            transform = lambda _metadata: borrowed
        if fuse12 and local_tokens >= _STAGE12_MIN_M:
            from aiter.ops.flydsl.mxfp4_kname import parse_flydsl_v2_gemm2_kernel

            stage12_transform, box = self._fused_stage12(
                plan,
                local_tokens,
                self._collectives(plan.ag_wire),
                parse_flydsl_v2_gemm2_kernel(plan.gemm2_kernel),
                base_transform=transform,
                tk_ids=topk_ids_all,
                tk_weights=topk_weights_all,
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
            transform = lambda md: md
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
        return partial, (box[0] if box is not None else None)

    def sort(self, topk_weights_all, topk_ids_all, plan: "_CasePlan"):
        """Run the expert sort as its own step, ahead of the fused region."""
        from aiter.fused_moe import moe_sorting

        cfg = self.cfg
        metadata = plan.metadata
        if metadata is None:
            raise ValueError("the hoisted sort needs a resolved tuned row")
        total = int(topk_ids_all.shape[0])
        accumulate = self._plan_accumulate(plan)
        if metadata.output_aux:
            return moe_sorting(
                topk_ids_all,
                topk_weights_all,
                cfg.experts,
                cfg.model_dim,
                dtypes.bf16,
                metadata.block_m,
                accumulate=accumulate,
                output_aux=metadata.output_aux,
                output=self._collectives(plan.ag_wire).partial_buffer(total),
            )
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid, moe_buf = moe_sorting(
            topk_ids_all,
            topk_weights_all,
            cfg.experts,
            cfg.model_dim,
            dtypes.bf16,
            metadata.block_m,
            accumulate=accumulate,
            flat=getattr(metadata, "flat", False),
            output=self._collectives(plan.ag_wire).partial_buffer(total),
        )
        return (
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid,
            moe_buf,
            None,
            None,
        )

    @staticmethod
    def _plan_accumulate(plan: "_CasePlan") -> bool:
        """Whether GEMM2's epilogue accumulates into the sort's output buffer."""
        from aiter.fused_moe import stage2_uses_route_reduce

        return not stage2_uses_route_reduce(plan.metadata.stage2)

    def reduce_scatter(self, local_tokens: int, plan: "_CasePlan"):
        """Sum the per-rank partials across TP and keep this rank's token shard."""
        return self._collectives(plan.ag_wire).reduce_scatter(local_tokens)

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
        if _TIME_STEPS and not torch.cuda.is_current_stream_capturing():
            return self._forward_timed(x_local, topk_weights, topk_ids, plan, m)
        a1, a1_scale, wts, ids = self.all_gather(
            x_local, topk_weights, topk_ids, plan=plan
        )
        _, y_local = self.local_moe(a1, a1_scale, wts, ids, plan=plan, local_tokens=m)
        if y_local is not None:
            return y_local
        return self.reduce_scatter(m, plan)

    _sort_events: "list" = []

    @staticmethod
    def _install_sort_probe():
        from aiter import fused_moe as _fm

        if getattr(_fm.moe_sorting, "_mega_timed", False):
            return

        original = _fm.moe_sorting

        def timed(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            out = original(*args, **kwargs)
            stop.record()
            MegaMoeTP._sort_events.append((start, stop))
            return out

        timed._mega_timed = True
        _fm.moe_sorting = timed

    def _forward_timed(self, x_local, topk_weights, topk_ids, plan, m):
        self._install_sort_probe()
        MegaMoeTP._sort_events.clear()
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        ev[0].record()
        a1, a1_scale, wts, ids = self.all_gather(
            x_local, topk_weights, topk_ids, plan=plan
        )
        ev[1].record()
        _, y_local = self.local_moe(a1, a1_scale, wts, ids, plan=plan, local_tokens=m)
        ev[2].record()
        out = y_local if y_local is not None else self.reduce_scatter(m, plan)
        ev[3].record()
        torch.cuda.synchronize()
        self._step_us = tuple(
            ev[i].elapsed_time(ev[i + 1]) * 1e3 for i in range(3)
        )
        seen = getattr(self, "_step_seen", 0)
        self._step_seen = seen + 1
        if seen and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            ag, moe, rs = self._step_us
            sub = getattr(self, "_ag_sub_us", None)
            detail = (
                "" if sub is None else f" [route={sub[0]:7.1f} pub={sub[1]:7.1f}]"
            )
            sort = sum(
                a.elapsed_time(b) * 1e3 for a, b in MegaMoeTP._sort_events
            )
            print(
                f"[STEP] m={m} ag={ag:8.1f} local_moe={moe:8.1f} "
                f"rs={rs:8.1f} total={ag + moe + rs:8.1f}{detail}"
                f" sort={sort:8.1f} kern={moe - sort:8.1f}",
                flush=True,
            )
        return out

    __call__ = forward


@dataclass(frozen=True)
class _CasePlan:
    """The per-token-bucket decisions the runtime caches."""

    tokens: int
    ag_wire: str
    gemm1_kernel: str
    gemm2_kernel: str
    metadata: object | None = None
    gemm_bucket: int = 0
    fuse_rs: bool = False
    waves_per_eu: int = 0
    stage12: bool = True


def _prequant_transform(metadata):
    """Force the pre-quantized activation path for an FP4 wire."""
    return replace(metadata, prequant=True)
