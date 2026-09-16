# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Typed problem, candidate, result, and CSV contracts for MHA forward tuning."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from typing import Any, Literal, Mapping

MHA_FWD_HARDWARE_KEY_FIELDS = ("gfx", "gpu_model", "cu_num")
MHA_FWD_PROBLEM_KEY_FIELDS = (
    "mode",
    "batch",
    "total_q",
    "total_k",
    "max_seqlen_q",
    "max_seqlen_k",
    "min_seqlen_q",
    "nhead_q",
    "nhead_k",
    "hdim_q",
    "hdim_v",
    "dtype",
    "causal",
    "window_left",
    "window_right",
    "sink_size",
    "dropout_p",
    "logits_soft_cap",
    "how_v3_bf16_cvt",
    "return_lse",
    "return_attn_probs",
    "has_bias",
    "has_alibi",
    "has_sink",
    "has_block_table",
    "has_q_descale",
    "has_physical_padding",
    "is_grad",
)
MHA_FWD_TUNING_KEY_FIELDS = (
    *MHA_FWD_HARDWARE_KEY_FIELDS,
    *MHA_FWD_PROBLEM_KEY_FIELDS,
)
MHA_FWD_CANDIDATE_FIELDS = ("backend", "num_splits", "backend_config")
MHA_FWD_RUNTIME_SELECTION_FIELDS = ("backend", "num_splits", "backend_config")
MHA_FWD_METRIC_FIELDS = (
    "us",
    "errRatio",
    "status",
    "detail",
    "samples_us",
    "tflops",
)
MHA_FWD_RUNTIME_CSV_FIELDS = (
    *MHA_FWD_TUNING_KEY_FIELDS,
    *MHA_FWD_RUNTIME_SELECTION_FIELDS,
)
MHA_FWD_MEASUREMENT_CSV_FIELDS = (
    *MHA_FWD_TUNING_KEY_FIELDS,
    *MHA_FWD_CANDIDATE_FIELDS,
    *MHA_FWD_METRIC_FIELDS,
)

MhaFwdBackend = Literal["asm_v3", "ck", "flydsl", "gluon", "opus", "triton"]
MhaFwdStatus = Literal[
    "ok",
    "unsupported",
    "mismatch",
    "oom_preflight",
    "oom_runtime",
    "timeout",
    "crash",
]
MhaFwdRunState = Literal[
    "failed",
    "partial",
    "measured",
    "verified",
    "review-ready",
]

MHA_FWD_BACKENDS = frozenset(
    {"asm_v3", "ck", "flydsl", "gluon", "opus", "triton"}
)
MHA_FWD_RUNTIME_BACKENDS = MHA_FWD_BACKENDS
MHA_FWD_RESULT_STATUSES = frozenset(
    {
        "ok",
        "unsupported",
        "mismatch",
        "oom_preflight",
        "oom_runtime",
        "timeout",
        "crash",
    }
)
MHA_FWD_RUN_STATES = frozenset(
    {"failed", "partial", "measured", "verified", "review-ready"}
)


def csv_scalar(value: Any) -> str:
    """Normalize one native CSV key value to stable Aiter spelling."""

    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return str(float(value))
    return str(value).strip()


def canonical_backend_config(config: Mapping[str, Any] | None) -> str:
    """Return deterministic JSON for one backend launch configuration."""

    return (
        json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
        if config
        else ""
    )


def parse_backend_config(value: Any) -> dict[str, Any] | None:
    """Parse a runtime CSV backend_config cell into a mapping."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("backend_config must be a JSON object")
    return parsed


def normalize_mha_dtype(value: Any) -> str:
    normalized = csv_scalar(value).removeprefix("torch.").lower()
    return {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }.get(normalized, normalized)


@dataclass(frozen=True, slots=True)
class MhaFwdProblem:
    gfx: str
    gpu_model: str
    cu_num: int
    mode: str
    batch: int
    total_q: int
    total_k: int
    max_seqlen_q: int
    max_seqlen_k: int
    min_seqlen_q: int
    nhead_q: int
    nhead_k: int
    hdim_q: int
    hdim_v: int
    dtype: str
    causal: bool
    window_left: int
    window_right: int
    sink_size: int
    dropout_p: float
    logits_soft_cap: float
    how_v3_bf16_cvt: int
    return_lse: bool
    return_attn_probs: bool
    has_bias: bool
    has_alibi: bool
    has_sink: bool
    has_block_table: bool
    has_q_descale: bool
    has_physical_padding: bool
    is_grad: bool

    def __post_init__(self) -> None:
        if not self.gfx.startswith("gfx"):
            raise ValueError(f"invalid MHA architecture {self.gfx!r}")
        if not self.gpu_model or self.gpu_model == "unknown":
            raise ValueError("gpu_model must identify the measured GPU SKU")
        if self.cu_num <= 0:
            raise ValueError("cu_num must be positive")
        if self.mode not in ("batch", "varlen"):
            raise ValueError(f"unsupported MHA mode {self.mode!r}")
        for name in (
            "batch",
            "total_q",
            "total_k",
            "max_seqlen_q",
            "max_seqlen_k",
            "nhead_q",
            "nhead_k",
            "hdim_q",
            "hdim_v",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.nhead_q % self.nhead_k:
            raise ValueError("nhead_q must be divisible by nhead_k")
        if not self.max_seqlen_q <= self.total_q <= self.batch * self.max_seqlen_q:
            raise ValueError("total_q is inconsistent with batch and max_seqlen_q")
        if not self.max_seqlen_k <= self.total_k <= self.batch * self.max_seqlen_k:
            raise ValueError("total_k is inconsistent with batch and max_seqlen_k")
        if not 0 <= self.min_seqlen_q <= self.max_seqlen_q:
            raise ValueError("min_seqlen_q must be in [0, max_seqlen_q]")
        if not 0.0 <= self.dropout_p <= 1.0:
            raise ValueError("dropout_p must be in [0, 1]")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> MhaFwdProblem:
        missing = [field for field in MHA_FWD_TUNING_KEY_FIELDS if field not in row]
        if missing:
            raise ValueError(f"MHA problem is missing fields: {missing}")
        return cls(
            gfx=csv_scalar(row["gfx"]).lower(),
            gpu_model=csv_scalar(row["gpu_model"]).lower(),
            cu_num=int(row["cu_num"]),
            mode=csv_scalar(row["mode"]).lower(),
            batch=int(row["batch"]),
            total_q=int(row["total_q"]),
            total_k=int(row["total_k"]),
            max_seqlen_q=int(row["max_seqlen_q"]),
            max_seqlen_k=int(row["max_seqlen_k"]),
            min_seqlen_q=int(row["min_seqlen_q"]),
            nhead_q=int(row["nhead_q"]),
            nhead_k=int(row["nhead_k"]),
            hdim_q=int(row["hdim_q"]),
            hdim_v=int(row["hdim_v"]),
            dtype=normalize_mha_dtype(row["dtype"]),
            causal=_as_bool(row["causal"]),
            window_left=int(row["window_left"]),
            window_right=int(row["window_right"]),
            sink_size=int(row["sink_size"]),
            dropout_p=float(row["dropout_p"]),
            logits_soft_cap=float(row["logits_soft_cap"]),
            how_v3_bf16_cvt=int(row["how_v3_bf16_cvt"]),
            return_lse=_as_bool(row["return_lse"]),
            return_attn_probs=_as_bool(row["return_attn_probs"]),
            has_bias=_as_bool(row["has_bias"]),
            has_alibi=_as_bool(row["has_alibi"]),
            has_sink=_as_bool(row["has_sink"]),
            has_block_table=_as_bool(row["has_block_table"]),
            has_q_descale=_as_bool(row["has_q_descale"]),
            has_physical_padding=_as_bool(row["has_physical_padding"]),
            is_grad=_as_bool(row["is_grad"]),
        )

    def key(self) -> tuple[str, ...]:
        return tuple(csv_scalar(getattr(self, field)) for field in MHA_FWD_TUNING_KEY_FIELDS)

    def as_row(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in MHA_FWD_TUNING_KEY_FIELDS}


@dataclass(frozen=True, slots=True)
class MhaFwdCandidate:
    backend: MhaFwdBackend
    num_splits: int = 0
    backend_config: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        validate_mha_fwd_plan_fields(
            self.backend, self.num_splits, self.backend_config
        )

    @property
    def config_json(self) -> str:
        return canonical_backend_config(self.backend_config)

    @property
    def identity(self) -> tuple[str, int, str]:
        return self.backend, self.num_splits, self.config_json


@dataclass(frozen=True, slots=True)
class MhaFwdPlan:
    backend: MhaFwdBackend
    num_splits: int = 0
    backend_config: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        validate_mha_fwd_plan_fields(
            self.backend, self.num_splits, self.backend_config
        )

    def validate_for(self, problem: MhaFwdProblem) -> None:
        validate_mha_fwd_backend_arch(self.backend, problem.gfx)
        if self.backend == "asm_v3" and (
            problem.gfx != "gfx942"
            or problem.mode != "varlen"
            or problem.dtype != "bfloat16"
            or problem.hdim_q != 192
            or problem.hdim_v != 128
            or problem.dropout_p != 0.0
            or problem.logits_soft_cap != 0.0
            or problem.window_left > 0
            or problem.window_right > 0
            or problem.sink_size != 0
            or problem.return_attn_probs
            or problem.has_bias
            or problem.has_alibi
            or problem.has_sink
            or problem.has_block_table
            or problem.has_q_descale
            or problem.is_grad
        ):
            raise ValueError(
                "ASM split policy requires the compatible gfx942 packed-varlen "
                "bf16 D_QK=192/D_V=128 inference path"
            )


@dataclass(frozen=True, slots=True)
class MhaFwdResult:
    problem: MhaFwdProblem
    candidate: MhaFwdCandidate
    status: MhaFwdStatus
    err_ratio: float
    samples_us: tuple[float, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in MHA_FWD_RESULT_STATUSES:
            raise ValueError(f"unsupported MHA result status {self.status!r}")
        if not math.isfinite(self.err_ratio) or self.err_ratio < 0:
            raise ValueError("err_ratio must be finite and non-negative")
        if any(not math.isfinite(sample) or sample <= 0 for sample in self.samples_us):
            raise ValueError("timing samples must be finite and positive")
        if self.status == "ok" and not self.samples_us:
            raise ValueError("successful MHA results require timing samples")

    @property
    def median_us(self) -> float:
        return (
            float(statistics.median(self.samples_us))
            if self.samples_us
            else float("inf")
        )


def validate_mha_fwd_plan_fields(
    backend: str,
    num_splits: int,
    backend_config: Mapping[str, Any] | None,
) -> None:
    if backend not in MHA_FWD_BACKENDS:
        raise ValueError(f"unsupported MHA backend {backend!r}")
    if not 0 <= int(num_splits) <= 8:
        raise ValueError("num_splits must be in [0, 8]")
    if backend == "asm_v3":
        if int(num_splits) < 1:
            raise ValueError("asm_v3 requires an explicit split count in [1, 8]")
    elif int(num_splits) != 0:
        raise ValueError(f"{backend} does not accept an external split count")
    if backend_config and backend not in ("triton", "gluon"):
        raise ValueError(f"{backend} does not accept backend_config")


def validate_mha_fwd_backend_arch(backend: str, gfx: str) -> None:
    supported = {
        "asm_v3": {"gfx942", "gfx950"},
        "ck": {"gfx942", "gfx950", "gfx1250"},
        "flydsl": {"gfx1250"},
        "gluon": {"gfx950"},
        "opus": {"gfx950"},
        "triton": {"gfx942", "gfx950", "gfx1250"},
    }
    if gfx not in supported[backend]:
        raise ValueError(f"MHA backend {backend!r} does not support {gfx!r}")


def enumerate_mha_fwd_candidates(gfx: str) -> tuple[MhaFwdCandidate, ...]:
    """Return the complete legal offline search catalogue for one architecture."""

    candidates: list[MhaFwdCandidate] = []
    if gfx in ("gfx942", "gfx950"):
        candidates.extend(MhaFwdCandidate("asm_v3", split) for split in range(1, 9))
    candidates.append(MhaFwdCandidate("ck"))

    triton_axes = (
        (16, 32, 64, 128, 256),
        (16, 32, 64, 128),
        (False, True),
        (2, 4, 8),
        (1, 2, 3, 4),
        (1, 2, 3),
    )
    for block_m, block_n, preload_v, warps, waves, stages in product(*triton_axes):
        candidates.append(
            MhaFwdCandidate(
                "triton",
                backend_config={
                    "BLOCK_M": block_m,
                    "BLOCK_N": block_n,
                    "PRELOAD_V": preload_v,
                    "num_warps": warps,
                    "waves_per_eu": waves,
                    "num_stages": stages,
                    "num_ctas": 1,
                },
            )
        )

    if gfx == "gfx950":
        gluon_axes = (
            (16, 32, 64, 128, 256),
            (32, 64, 128),
            (2, 4, 8),
            (1, 2, 3, 4),
        )
        for block_m, block_n, warps, waves in product(*gluon_axes):
            candidates.append(
                MhaFwdCandidate(
                    "gluon",
                    backend_config={
                        "BLOCK_M": block_m,
                        "BLOCK_N": block_n,
                        "num_warps": warps,
                        "waves_per_eu": waves,
                    },
                )
            )
        candidates.append(MhaFwdCandidate("opus"))
    if gfx == "gfx1250":
        candidates.append(MhaFwdCandidate("flydsl"))

    identities = [candidate.identity for candidate in candidates]
    if len(identities) != len(set(identities)):
        raise RuntimeError(f"duplicate MHA candidate identity for {gfx}")
    return tuple(candidates)


def mha_fwd_candidate_id(
    problem: MhaFwdProblem, candidate: MhaFwdCandidate
) -> str:
    """Return a stable identifier used by checkpoint journals and resume."""

    payload = json.dumps(
        {"problem": problem.key(), "candidate": candidate.identity},
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"invalid boolean value {value!r}")
