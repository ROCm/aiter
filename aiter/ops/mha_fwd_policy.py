# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MHA forward tuned rows: the schema shared with the tuner, and the runtime lookup.

A row pins the backend for one exact problem on one GPU, never a backend's own
tile. Dispatch's own gates still decide whether that backend can serve a call;
a row it cannot honor falls back to auto-select.

Triton and Gluon tiles live in Triton's JSON configs, so their rows carry
``config_digest``, the hash of the tile Triton resolved at tuning time, and are
dropped once Triton resolves another one.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import torch

from .. import logger
from ..jit.core import AITER_CONFIGS, AITER_LOG_TUNED_CONFIG, ENABLE_CK
from ..jit.utils.chip_info import get_gfx, get_gfx_runtime, get_gpu_model
from ..utility.dispatch_verification import record_selection

# The hardware a tuned row is valid for, in tuned-CSV column order.
TUNING_HARDWARE_FIELDS = ("gfx", "gpu_model", "cu_num")


@functools.lru_cache(maxsize=8)
def get_tuning_hardware(device_id: int = 0) -> dict[str, str | int]:
    """Cached and shared: do not mutate."""
    return {
        "gfx": get_gfx_runtime(),
        "gpu_model": get_gpu_model(device_id),
        "cu_num": torch.cuda.get_device_properties(device_id).multi_processor_count,
    }


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
# Dispatch reads backend; us is kept for people, and config_digest is only set
# for the Triton backends.
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
MHA_FWD_DTYPES = frozenset(
    {
        "bfloat16",
        "float16",
        "float32",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
    }
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


# Field annotations are strings under `from __future__ import annotations`.
_CELL_PARSERS = {
    "str": lambda value: csv_scalar(value).lower(),
    "int": int,
    "float": float,
    "bool": as_bool,
}


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
        if self.mode != "varlen":
            raise ValueError(f"unsupported MHA mode {self.mode!r}")
        # Their row counts are physical rows and KV pool blocks, not tokens.
        if self.has_physical_padding or self.has_block_table:
            raise ValueError("rows for padded or paged calls are not supported")
        if self.dtype not in MHA_FWD_DTYPES:
            raise ValueError(f"unsupported MHA dtype {self.dtype!r}")
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
            **{
                f.name: (
                    normalize_mha_dtype if f.name == "dtype" else _CELL_PARSERS[f.type]
                )(row[f.name])
                for f in fields(cls)
            }
        )

    def key(self) -> tuple[str, ...]:
        return tuple(
            csv_scalar(getattr(self, field)) for field in MHA_FWD_TUNING_KEY_FIELDS
        )


def triton_unsupported_reason(problem: MhaFwdProblem, backend: str) -> str | None:
    """Why Triton's varlen entry cannot stand in for the CK path here, else None.

    On a CK build the caller expects what the CK path returns, not just the
    same attention.
    """
    if problem.logits_soft_cap:
        return "Triton MHA has no logits soft-cap"
    if problem.has_q_descale:
        return "prequantized FP8 attention stays on CK"
    if problem.sink_size:
        return "Triton MHA has no sink tokens in the window"
    if problem.return_lse or problem.return_attn_probs:
        return "Triton returns LSE and probabilities in another layout than CK"
    if problem.is_grad:
        return "a forward-tuned row must not move the backward pass to Triton"
    if problem.window_right != -1:
        return "Triton MHA has no right window"
    if problem.has_bias:
        return "Triton MHA has no bias"
    if problem.min_seqlen_q:
        return "the Triton varlen entry ignores min_seqlen_q, which CK honours"
    if problem.window_left == 0:
        return "Triton reads a left window of 0 as no window"
    if problem.has_alibi or problem.has_sink:
        return "no tuner run measures ALiBi or sink logits on Triton"
    if backend != "gluon":
        return None
    from .triton.attention.mha import gluon_forward_unsupported_reason

    return gluon_forward_unsupported_reason(dropout_p=problem.dropout_p)


def triton_config(problem: MhaFwdProblem, backend: str) -> dict[str, Any]:
    """The tile Triton's own resolver picks for this problem on this GPU.

    Mirrors the arguments the Triton varlen entry passes when ``config`` is None.
    """
    has_pe = problem.hdim_q > problem.hdim_v
    dtype = getattr(torch, problem.dtype)
    if backend == "gluon":
        from .triton._gluon_kernels.gfx950.attention.mha import _get_config
        from .triton.utils import types

        is_fp8 = types._is_fp8(torch.empty(0, dtype=dtype))
        return dict(_get_config(is_fp8=is_fp8, has_pe=has_pe))

    from .triton._triton_kernels.attention.mha import _get_config

    return dict(
        _get_config(
            problem.dropout_p > 0.0,
            dtype,
            has_pe=has_pe,
            head_dim_v=problem.hdim_v,
            causal=problem.causal,
            max_seqlen_q=problem.max_seqlen_q,
            max_seqlen_k=problem.max_seqlen_k,
            batch_heads=problem.batch * problem.nhead_q,
        )
    )


def config_digest(config: Mapping[str, Any]) -> str:
    # Prefixed so it never reads as a number: a CSV merge without dtype=str
    # would turn an all-digit hex digest into an int or a float.
    payload = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def _triton_row_rejection(row: dict, problem: MhaFwdProblem) -> str | None:
    """Why a Triton or Gluon row must not be honored on this GPU, else None."""
    backend = row["backend"].strip()
    reason = triton_unsupported_reason(problem, backend)
    if reason is not None:
        return reason
    expected = (row.get("config_digest") or "").strip()
    if not expected:
        return f"a {backend} row needs config_digest"
    actual = config_digest(triton_config(problem, backend))
    if actual != expected:
        return (
            f"Triton now resolves another {backend} config for this shape "
            f"(digest {actual}, row tuned with {expected}); retune it with "
            f"{MHA_FWD_TUNER_SCRIPT}"
        )
    return None


@functools.lru_cache(maxsize=8)
def _load_mha_fwd_tuning_table(path: str) -> dict[tuple[str, ...], str]:
    """Exact key -> backend.

    A bad row is skipped with a warning, so one row cannot break every MHA call.
    Only Triton and Gluon rows for this GPU's arch are checked against Triton.
    """
    table: dict[tuple[str, ...], str] = {}
    if not os.path.isfile(path):
        if os.environ.get(MHA_FWD_CONFIG_ENV):
            logger.warning(
                "%s=%s does not exist; MHA forward runs untuned",
                MHA_FWD_CONFIG_ENV,
                path,
            )
        return table
    with open(path, encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = [
            field
            for field in (*MHA_FWD_TUNING_KEY_FIELDS, "backend")
            if field not in (reader.fieldnames or ())
        ]
        if missing:
            logger.warning("%s is missing MHA columns %s; ignoring it", path, missing)
            return table
        for line, row in enumerate(reader, start=2):
            try:
                backend = str(row["backend"]).strip()
                if backend not in MHA_FWD_BACKENDS:
                    raise ValueError(f"unknown backend {backend!r}")
                problem = MhaFwdProblem.from_mapping(row)
                if backend in MHA_FWD_TRITON_BACKENDS and problem.gfx == get_gfx():
                    reason = _triton_row_rejection(row, problem)
                    if reason is not None:
                        raise ValueError(reason)
                key = problem.key()
            # Any error: the digest check imports and runs Triton's resolver.
            except Exception as error:  # noqa: BLE001
                logger.warning("%s:%d: skipping MHA tuned row: %s", path, line, error)
                continue
            if key in table:
                logger.warning(
                    "%s:%d: duplicate MHA key; keeping the first", path, line
                )
                continue
            table[key] = backend
    return table


@functools.lru_cache(maxsize=1)
def _mha_fwd_table_path() -> str:
    # Resolving the property globs and merges model_configs, so do it once.
    return os.path.abspath(AITER_CONFIGS.AITER_CONFIG_MHA_FWD_FILE)


_COARSE_FIELDS = ("mode", "hdim_q", "hdim_v", "dtype")


@functools.lru_cache(maxsize=8)
def _mha_fwd_hardware_prefixes(path: str) -> frozenset[tuple[str, ...]]:
    span = len(TUNING_HARDWARE_FIELDS)
    return frozenset(key[:span] for key in _load_mha_fwd_tuning_table(path))


@functools.lru_cache(maxsize=8)
def _mha_fwd_coarse_keys(path: str) -> frozenset[tuple[str, ...]]:
    span = len(TUNING_HARDWARE_FIELDS)
    positions = [MHA_FWD_TUNING_KEY_FIELDS.index(f) for f in _COARSE_FIELDS]
    return frozenset(
        (*key[:span], *(key[i] for i in positions))
        for key in _load_mha_fwd_tuning_table(path)
    )


def clear_mha_fwd_tuning_cache() -> None:
    """Re-read the tuned table (and its path) on the next call."""
    _mha_fwd_table_path.cache_clear()
    _load_mha_fwd_tuning_table.cache_clear()
    _mha_fwd_hardware_prefixes.cache_clear()
    _mha_fwd_coarse_keys.cache_clear()


def _mha_fwd_has_rows(prefix: tuple[str, ...] | None = None) -> bool:
    """Whether the table has rows at all, or for one (gfx, gpu_model, cu_num)."""
    prefixes = _mha_fwd_hardware_prefixes(_mha_fwd_table_path())
    return bool(prefixes) if prefix is None else prefix in prefixes


def _mha_fwd_has_coarse_rows(coarse: tuple[str, ...]) -> bool:
    return coarse in _mha_fwd_coarse_keys(_mha_fwd_table_path())


def _mha_fwd_local_hardware(device_id: int) -> tuple[str, ...]:
    """This device's (gfx, gpu_model, cu_num) as row cells, or () when the
    running arch is not the built one, which the kernel gates test."""
    hardware = get_tuning_hardware(device_id)
    if hardware["gfx"] != get_gfx():
        return ()
    return tuple(csv_scalar(hardware[field]) for field in TUNING_HARDWARE_FIELDS)


def _mha_fwd_backend_for_key(key: tuple[str, ...]) -> str | None:
    path = _mha_fwd_table_path()
    backend = _load_mha_fwd_tuning_table(path).get(key)
    if backend is not None and AITER_LOG_TUNED_CONFIG:
        logger.info("MHA fwd matched a tuned row in %s: %s", path, backend)
    return backend


def _mha_fwd_tuning_key(
    hardware: tuple[str, ...],
    *,
    mode: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    min_seqlen_q: int,
    causal: bool,
    window_size: tuple[int, ...],
    dropout_p: float,
    logits_soft_cap: float,
    how_v3_bf16_cvt: int,
    return_lse: bool,
    return_attn_probs: bool,
    bias: torch.Tensor | None,
    alibi_slopes: torch.Tensor | None,
    sink_ptr: torch.Tensor | None,
    block_table: torch.Tensor | None,
    q_descale: torch.Tensor | None,
    cu_seqlens_q_padded: torch.Tensor | None,
    cu_seqlens_k_padded: torch.Tensor | None,
) -> tuple[str, ...]:
    # Coerced as MhaFwdProblem coerces row cells, so dropout_p=0 and 0.0, or
    # causal=1 and True, find the same row.
    values = {
        **dict(zip(TUNING_HARDWARE_FIELDS, hardware)),
        "mode": mode,
        "batch": int(batch),
        "total_q": q.shape[0],
        "total_k": k.shape[0],
        "max_seqlen_q": int(max_seqlen_q),
        "max_seqlen_k": int(max_seqlen_k),
        "min_seqlen_q": int(min_seqlen_q),
        "nhead_q": q.shape[-2],
        "nhead_k": k.shape[-2],
        "hdim_q": q.shape[-1],
        "hdim_v": v.shape[-1],
        "dtype": str(q.dtype).removeprefix("torch."),
        "causal": bool(causal),
        "window_left": int(window_size[0]),
        "window_right": int(window_size[1]),
        "sink_size": int(window_size[2]) if len(window_size) > 2 else 0,
        "dropout_p": float(dropout_p),
        "logits_soft_cap": float(logits_soft_cap),
        "how_v3_bf16_cvt": int(how_v3_bf16_cvt),
        "return_lse": bool(return_lse),
        "return_attn_probs": bool(return_attn_probs),
        "has_bias": bias is not None,
        "has_alibi": alibi_slopes is not None,
        "has_sink": sink_ptr is not None,
        "has_block_table": block_table is not None,
        "has_q_descale": q_descale is not None,
        "has_physical_padding": (
            cu_seqlens_q_padded is not None or cu_seqlens_k_padded is not None
        ),
        "is_grad": torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)),
    }
    return tuple(csv_scalar(values[field]) for field in MHA_FWD_TUNING_KEY_FIELDS)


def get_mha_fwd_tuned_backend(**key_args) -> str | None:
    """The backend a tuned row pins for this call, or None to auto-select.

    Rows only apply to CK-enabled builds; without CK, dispatch goes to Triton,
    which keeps its own configs. Code dynamo traces auto-selects: the key
    reads sizes that dynamo makes symbolic, which breaks the graph and
    recompiles per token count. Graph capture of an eager or opaque caller
    still runs the row.
    """
    if torch.compiler.is_compiling():
        return None
    if not ENABLE_CK or not _mha_fwd_has_rows():
        return None
    if key_args["block_table"] is not None or not (
        key_args["cu_seqlens_q_padded"] is None
        and key_args["cu_seqlens_k_padded"] is None
    ):
        return None
    q, v = key_args["q"], key_args["v"]
    prefix = _mha_fwd_local_hardware(
        q.device.index if q.device.index is not None else 0
    )
    if not prefix or not _mha_fwd_has_rows(prefix):
        return None
    coarse = (
        *prefix,
        key_args["mode"],
        str(q.shape[-1]),
        str(v.shape[-1]),
        str(q.dtype).removeprefix("torch."),
    )
    if not _mha_fwd_has_coarse_rows(coarse):
        return None
    return _mha_fwd_backend_for_key(_mha_fwd_tuning_key(prefix, **key_args))


@functools.cache
def warn_declined_mha_fwd_row(backend: str, **shape: int) -> None:
    """Once per backend and shape; ``shape`` takes ints only, as it keys the cache."""
    logger.warning(
        "a tuned MHA row pins %r for a call it cannot serve (%s); using auto-select without FlyDSL",
        backend,
        ", ".join(f"{name}={value}" for name, value in shape.items()),
    )


def record_mha_fwd_selection(backend: str) -> None:
    record_selection(MHA_FWD_FAMILY, backend=backend)
