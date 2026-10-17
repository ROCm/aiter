# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Canonical semantic description of a GEMM followed by an epilogue chain.

Pure Python (no torch import). Every parameter that affects output bytes is a
dataclass field and therefore part of the signature.

Canonicalization rules (SCHEMA_VERSION = 1), implemented by ``canonical()``:
  * dataclass -> JSON object {"_t": <class name>, <field>: <canonical value>...}
    (the ``_t`` tag keeps different stage types with equal fields apart);
  * tuple/list -> JSON array, order preserved (field order is semantic);
  * int -> JSON integer; bool -> true/false (bool is never confused with int);
  * float -> JSON string "f:" + repr(float(x)) (shortest round-trip repr, so
    1e-6 == 0.000001 == float("1e-06"); NaN/inf are rejected);
  * str -> JSON string; None -> null;
  * object keys sorted, separators (",", ":"), ensure_ascii=True;
  * signature = sha256(utf-8 json)[:16] hex; the blob embeds SCHEMA_VERSION.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from typing import Optional, Tuple, Union

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GemmSpec:
    quant: str = "fp8_ptpc"  # per-token act scale x per-channel weight scale
    a_dtype: str = "fp8_e4m3fn"
    w_dtype: str = "fp8_e4m3fn"
    w_layout: str = "shuffle_16x16"
    acc: str = "fp32"
    # scale order and single rounding are part of the contract
    epilogue_math: str = "bf16_rne((acc*ws)*as)"
    n: int = 0
    k: int = 0


@dataclass(frozen=True)
class Field:
    """A column range of the GEMM output."""

    name: str  # "q" | "k" | "v" | "iq" | "ik"
    offset: int
    width: int
    shards: int  # = world: scatter heads; 1: replicate


@dataclass(frozen=True)
class SPExchange:
    """Ulysses token<->head exchange."""

    kind: str = "sp_exchange"
    world: int = 4
    fields: Tuple[Field, ...] = ()
    drop: Tuple[str, ...] = ()  # fields not consumed downstream
    out_layout: str = "packed_rows"


@dataclass(frozen=True)
class HeadNormRopeCache:
    """Per-head RMSNorm + partial NeoX RoPE + KV/index cache writes."""

    kind: str = "head_norm_rope_cache"
    head_dim: int = 128
    q_heads: int = 16
    kv_heads: int = 1
    rotary_dim: int = 64
    rope: str = "neox"
    eps: float = 1e-6
    norm: str = "gemma_rms"
    numerics: str = "triton_norm_rope_cache_v1"  # or "aiter_fused_qknorm_idxrqknorm_v1"
    kv_cache: str = "shuffle_page16_fp8_ptok"
    index: Optional[str] = None  # "fp8_unit_page128" or None


Stage = Union[SPExchange, HeadNormRopeCache]


def canonical(obj):
    """Return a JSON-serializable canonical form (see module docstring)."""
    if is_dataclass(obj) and not isinstance(obj, type):
        d = {"_t": type(obj).__name__}
        for f in fields(obj):
            d[f.name] = canonical(getattr(obj, f.name))
        return d
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("non-finite float in spec")
        return "f:" + repr(float(obj))
    if isinstance(obj, (tuple, list)):
        return [canonical(x) for x in obj]
    if obj is None or isinstance(obj, str):
        return obj
    raise TypeError(f"unsupported type in spec: {type(obj)!r}")


@dataclass(frozen=True)
class FusionSpec:
    gemm: GemmSpec
    stages: Tuple[Stage, ...] = ()

    def prefix(self, n: int) -> "FusionSpec":
        return FusionSpec(self.gemm, tuple(self.stages[:n]))

    def canonical_json(self) -> str:
        blob = {"schema": SCHEMA_VERSION, "spec": canonical(self)}
        return json.dumps(blob, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def signature(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:16]


def qkv_fields(q: int, kv: int, world: int, iq: int = 0, ik: int = 0, kv_shards=None, ik_shards: int = 1):
    """Standard GEMM column map: q | k | v | [iq | ik]; widths are full (unsharded)."""
    out, off = [], 0
    for name, w, sh in (("q", q, world), ("k", kv, kv_shards or world), ("v", kv, kv_shards or world), ("iq", iq, world), ("ik", ik, ik_shards)):
        if w:
            out.append(Field(name, off, w, sh))
            off += w
    return tuple(out)
