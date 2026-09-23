# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Typed problem, candidate, result, and CSV contracts for MHA forward tuning."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from typing import Any, Literal

from ..jit.utils.chip_info import TUNING_HARDWARE_FIELDS

# The family's identity, stated once for the tuner and the tuning-test tables.
MHA_FWD_FAMILY = "mha_fwd"
MHA_FWD_TUNER_SCRIPT = "op_tests/tuners/tune_mha_fwd.py"
MHA_FWD_CONFIG_ENV = "AITER_CONFIG_MHA_FWD"
MHA_FWD_CONFIG_PROPERTY = "AITER_CONFIG_MHA_FWD_FILE"
MHA_FWD_TUNED_CSV = "tuned_mha_fwd.csv"
MHA_FWD_UNTUNED_CSV = "untuned_mha_fwd.csv"

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
    *TUNING_HARDWARE_FIELDS,
    *MHA_FWD_PROBLEM_KEY_FIELDS,
)
MHA_FWD_CANDIDATE_FIELDS = ("backend", "num_splits", "backend_config")
MHA_FWD_METRIC_FIELDS = (
    "us",
    "errRatio",
    "status",
    "detail",
    "samples_us",
    "tflops",
)
# The winning candidate's latency rides along with the row it justifies, as it
# does in every other tuned CSV in this repo. Dispatch never reads it; it is
# there so a human can see at a glance that a row was measured on plausible
# hardware. The rest of MHA_FWD_METRIC_FIELDS stays in the evidence CSV.
MHA_FWD_RUNTIME_EVIDENCE_FIELDS = ("us",)
MHA_FWD_RUNTIME_CSV_FIELDS = (
    *MHA_FWD_TUNING_KEY_FIELDS,
    *MHA_FWD_CANDIDATE_FIELDS,
    *MHA_FWD_RUNTIME_EVIDENCE_FIELDS,
)
MhaFwdBackend = Literal["asm_v3", "ck", "flydsl", "gluon", "opus", "triton"]
MHA_FWD_BACKENDS = frozenset({"asm_v3", "ck", "flydsl", "gluon", "opus", "triton"})
MHA_FWD_TILE_CONFIG_BACKENDS = frozenset({"gluon", "triton"})
MHA_FWD_TILE_CONFIG_KEYS = {
    "triton": frozenset(
        {
            "BLOCK_M",
            "BLOCK_N",
            "PRELOAD_V",
            "num_warps",
            "waves_per_eu",
            "num_stages",
            "num_ctas",
        }
    ),
    "gluon": frozenset({"BLOCK_M", "BLOCK_N", "num_warps", "waves_per_eu"}),
}


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
        # ValueError, not TypeError: the CSV loader wraps it to add file:line.
        raise ValueError("backend_config must be a JSON object")  # noqa: TRY004
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
            causal=as_bool(row["causal"]),
            window_left=int(row["window_left"]),
            window_right=int(row["window_right"]),
            sink_size=int(row["sink_size"]),
            dropout_p=float(row["dropout_p"]),
            logits_soft_cap=float(row["logits_soft_cap"]),
            how_v3_bf16_cvt=int(row["how_v3_bf16_cvt"]),
            return_lse=as_bool(row["return_lse"]),
            return_attn_probs=as_bool(row["return_attn_probs"]),
            has_bias=as_bool(row["has_bias"]),
            has_alibi=as_bool(row["has_alibi"]),
            has_sink=as_bool(row["has_sink"]),
            has_block_table=as_bool(row["has_block_table"]),
            has_q_descale=as_bool(row["has_q_descale"]),
            has_physical_padding=as_bool(row["has_physical_padding"]),
            is_grad=as_bool(row["is_grad"]),
        )

    def key(self) -> tuple[str, ...]:
        return tuple(
            csv_scalar(getattr(self, field)) for field in MHA_FWD_TUNING_KEY_FIELDS
        )

    def as_row(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in MHA_FWD_TUNING_KEY_FIELDS}


@dataclass(frozen=True, slots=True)
class MhaFwdCandidate:
    backend: MhaFwdBackend
    num_splits: int = 0
    backend_config: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        validate_mha_fwd_plan_fields(self.backend, self.num_splits, self.backend_config)

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
        validate_mha_fwd_plan_fields(self.backend, self.num_splits, self.backend_config)

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
        if self.backend == "asm_v3" and self.num_splits >= HD192_SPLITKV_MIN_SPLITS:
            rejected = hd192_splitkv_rejections(problem)
            if rejected:
                raise ValueError(
                    f"num_splits={self.num_splits} needs the gfx942 hd192 "
                    f"split-KV kernel: {', '.join(rejected)}"
                )


# Below this the C++ selector leaves the KV loop unsplit, so the split-KV
# kernel's constraints do not apply. Mirrors kFmhaHd192SplitKvMinSplits in
# csrc/include/mha_fwd.h.
HD192_SPLITKV_MIN_SPLITS = 2


def hd192_splitkv_rejections(problem: MhaFwdProblem) -> tuple[str, ...]:
    """Reasons the gfx942 hd192 split-KV kernel cannot serve ``problem``.

    Mirrors ``splitkv_compatible`` in ``csrc/py_itfs_cu/asm_mha_varlen_fwd.cu``.
    The C++ guard only judges the arguments that reached it, and the entry
    point which accepts a split count supplies constants for the mask, the
    padding and the conversion mode, so every condition has to hold before a
    row may force a split.
    """

    checks = (
        (problem.gfx == "gfx942", "arch is not gfx942"),
        (not problem.gpu_model.startswith("mi308"), "MI308 has no split-KV kernel"),
        (problem.mode == "varlen", "mode is not varlen"),
        (problem.dtype == "bfloat16", "dtype is not bf16"),
        (problem.batch == 1, "batch is not 1"),
        (problem.total_q == problem.max_seqlen_q, "q is not one packed sequence"),
        (problem.total_k == problem.max_seqlen_k, "k is not one packed sequence"),
        (problem.nhead_q == problem.nhead_k, "GQA is unsupported"),
        (problem.hdim_q == 192 and problem.hdim_v == 128, "head dims are not 192/128"),
        (not problem.causal, "causal masking is unsupported"),
        (problem.window_left == -1, "a left window is unsupported"),
        (problem.window_right == -1, "a right window is unsupported"),
        (problem.dropout_p == 0.0, "dropout is unsupported"),
        (problem.logits_soft_cap == 0.0, "a logits soft cap is unsupported"),
        (not problem.has_bias, "bias is unsupported"),
        (not problem.has_alibi, "alibi is unsupported"),
        (not problem.has_block_table, "paged KV is unsupported"),
        (not problem.has_q_descale, "descaling is unsupported"),
        (not problem.return_attn_probs, "returning dropout randval is unsupported"),
        (not problem.has_physical_padding, "padded cu_seqlens are unsupported"),
        (problem.how_v3_bf16_cvt == 1, "how_v3_bf16_cvt is not 1"),
    )
    return tuple(reason for holds, reason in checks if not holds)


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
    if backend_config:
        if backend not in MHA_FWD_TILE_CONFIG_BACKENDS:
            raise ValueError(f"{backend} does not accept backend_config")
        unknown = sorted(set(backend_config) - MHA_FWD_TILE_CONFIG_KEYS[backend])
        if unknown:
            raise ValueError(f"{backend} backend_config has unknown keys: {unknown}")


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


# Multiples of the combined standard error a winner must clear before it
# displaces the configuration already in use. Two is the conventional ~95%
# two-sample separation; the point is that the bar is stated once and is
# visible in the evidence rather than implied by whichever candidate sorted
# first.
MHA_FWD_SIGNIFICANCE_SIGMA = 2.0

# The indifference zone the race uses, as a fraction of the leader's latency.
# This is a reproducibility threshold rather than a taste parameter: an
# unchanged configuration moves by roughly this much between sessions on this
# hardware, so resolving differences below it would be resolving differences
# that do not survive to the next run. Under the race strategy it is also the
# bar a challenger must clear to displace the incumbent, because a shrinking
# standard-error test and a fixed indifference zone disagree about what a tie
# is, and running both would let one promote what the other called settled.
MHA_FWD_INDIFFERENCE_DELTA = 0.02


def enumerate_mha_fwd_candidates(
    gfx: str,
    backends: Sequence[str] | None = None,
) -> tuple[MhaFwdCandidate, ...]:
    """Return the legal offline search catalogue for one architecture.

    ``backends`` restricts the catalogue to the named backends. This is a
    control for comparing storage contracts, not a tuning mode: a winner drawn
    from a restricted field is the fastest of what was allowed to run, not the
    fastest available, so its evidence records the restriction.
    """
    if backends is not None:
        unknown = sorted(set(backends) - MHA_FWD_BACKENDS)
        if unknown:
            raise ValueError(f"unknown MHA backends {unknown}")

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

    if backends is not None:
        allowed = set(backends)
        candidates = [
            candidate for candidate in candidates if candidate.backend in allowed
        ]

    identities = [candidate.identity for candidate in candidates]
    if len(identities) != len(set(identities)):
        raise RuntimeError(f"duplicate MHA candidate identity for {gfx}")
    return tuple(candidates)


def mha_fwd_candidate_id(problem: MhaFwdProblem, candidate: MhaFwdCandidate) -> str:
    """Return a stable identifier used by checkpoint journals and resume."""

    payload = json.dumps(
        {"problem": problem.key(), "candidate": candidate.identity},
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def as_bool(value: Any) -> bool:
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
