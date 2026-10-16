# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The MHA forward tuned-row schema, shared by the runtime reader, the tuner and
the tuning tests.

A row pins the backend for one exact problem on one GPU, never a backend's own
tile: each backend keeps its configs where its owners keep them. Which backend
can actually serve a call stays with dispatch's own gates; a row it cannot
honor falls back to auto-select.

Triton and Gluon tiles live in Triton's JSON configs. A row that pins one of
them carries ``config_digest``, the hash of the tile Triton resolved when the
row was tuned, and the reader drops the row once Triton resolves another one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..jit.utils.chip_info import TUNING_HARDWARE_FIELDS

MHA_FWD_FAMILY = "mha_fwd"
MHA_FWD_CONFIG_ENV = "AITER_CONFIG_MHA_FWD"
MHA_FWD_TUNED_CSV = "tuned_mha_fwd.csv"

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
# Dispatch reads backend; us is the winner's latency, kept for people, and
# config_digest is only set for the Triton backends.
MHA_FWD_RUNTIME_CSV_FIELDS = (
    *MHA_FWD_TUNING_KEY_FIELDS,
    "backend",
    "us",
    "config_digest",
)
MHA_FWD_TRITON_BACKENDS = frozenset({"triton", "gluon"})
MHA_FWD_BACKENDS = frozenset(
    {"asm_v3", "ck", "flydsl", "opus", *MHA_FWD_TRITON_BACKENDS}
)


def csv_scalar(value: Any) -> str:
    """Normalize one native CSV key value to stable Aiter spelling."""

    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return str(float(value))
    return str(value).strip()


def normalize_mha_dtype(value: Any) -> str:
    normalized = csv_scalar(value).removeprefix("torch.").lower()
    return {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }.get(normalized, normalized)


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


def triton_unsupported_reason(problem: MhaFwdProblem, backend: str) -> str | None:
    """Why Triton's varlen entry cannot stand in for the CK path here, else None.

    On a CK build the caller expects what the CK path returns, so a Triton row
    must also return the same thing, not just compute the same attention.
    """
    if problem.mode != "varlen":
        return "only the varlen entry dispatches to Triton"
    if problem.logits_soft_cap:
        return "Triton MHA has no logits soft-cap"
    if problem.has_physical_padding:
        return "Triton MHA has no padded cu_seqlens"
    if problem.has_q_descale:
        return "prequantized FP8 attention stays on CK"
    if problem.sink_size:
        return "Triton MHA has no sink tokens in the window"
    if problem.return_lse or problem.return_attn_probs:
        return "Triton returns LSE and probabilities in another layout than CK"
    if problem.is_grad:
        # Rows are tuned on the forward pass; the backward would move with it.
        return "a forward-tuned row must not move the backward pass to Triton"
    if backend != "gluon":
        return None
    if problem.window_right != -1:
        return "Gluon MHA has no right window"
    from .triton.attention.mha import gluon_forward_unsupported_reason

    return gluon_forward_unsupported_reason(
        dropout_p=problem.dropout_p,
        bias=True if problem.has_bias else None,
        alibi_slopes=True if problem.has_alibi else None,
        block_table=True if problem.has_block_table else None,
    )


def triton_config(problem: MhaFwdProblem, backend: str) -> dict[str, Any]:
    """The tile Triton's own resolver picks for this problem on this GPU.

    Mirrors the arguments the Triton varlen entry passes when ``config`` is
    None, so the result is what that call would run.
    """
    has_pe = problem.hdim_q > problem.hdim_v
    if backend == "gluon":
        from .triton._gluon_kernels.gfx950.attention.mha import _get_config

        return dict(_get_config(is_fp8=False, has_pe=has_pe))

    import torch

    from .triton._triton_kernels.attention.mha import _get_config

    return dict(
        _get_config(
            problem.dropout_p > 0.0,
            getattr(torch, problem.dtype),
            has_pe=has_pe,
            head_dim_v=problem.hdim_v,
            causal=problem.causal,
            max_seqlen_q=problem.max_seqlen_q,
            max_seqlen_k=problem.max_seqlen_k,
            batch_heads=problem.batch * problem.nhead_q,
        )
    )


def config_digest(config: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
