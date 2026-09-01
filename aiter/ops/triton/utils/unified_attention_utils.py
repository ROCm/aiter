# SPDX-License-Identifier: MIT
# Copyright (C) 2026-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unified-attention config loading.

Each section is a flat table of complete configs, keyed by the case it covers::

    {
      "schema": {"attn_2d": ["D", "Q", "SW", "DT"], "kv_split": ["DT", "BS"]},

      "attn_2d": {
        "D_LEQ_128.Q_LEQ_1.SW": {...},   # small head, decode, sliding window
        "D_LEQ_128.Q_LEQ_1":    {...},   # small head, decode
        "D_GEQ_512":            {...},   # large head
        "any":                  {...}
      },

      "reduce": {"num_warps": 2, "num_stages": 1, "waves_per_eu": 2}
    }

A key names only the axes that matter; the rest are left out. A section with no
axes is a bare config, like ``reduce`` above. ``schema`` lists the axes each
section uses -- bounds are not declared, the keys state them.

Key components, by axis kind:

===== ================= =========================================
kind  example           matches
===== ================= =========================================
num   ``D_LEQ_128``     head_size <= 128 (also ``D_GEQ_512``)
bool  ``SW``            sliding_window > 0
enum  ``DT_fp8_fp8``    (q, kv) dtypes, falling back through
                        ``DT_fp8_any`` and ``DT_any_fp8``
===== ================= =========================================

Lookup tries each axis's candidates in schema order -- LEQ ascending, GEQ
descending, then "any" -- and takes the first key that exists. The leftmost
axis is therefore the most significant, which is how head_size outranks
max_seqlen_q.
"""

import copy
import functools
import itertools

import torch
import triton

from aiter.ops.triton.utils.config_utils import (
    USE_LRU_CACHE,
    load_config_json,
    resolve_config_dir,
)
from aiter.ops.triton.utils.types import e4m3_dtype

_CONFIG_NAME = "UNIFIED-ATTENTION"
_OPS = ("attn_2d", "attn_3d", "reduce", "kv_split")

_SEP = "."

_AXIS_KIND = {
    "D": "num",  # head_size
    "Q": "num",  # max_seqlen_q; Q_LEQ_1 is decode
    "BS": "num",  # block_size, the KV page size
    "SW": "bool",  # sliding_window > 0
    "SKLT": "bool",  # max_seqlen_k < 2048
    "SHUF": "bool",  # shuffled_kv_cache
    "DT": "enum",  # (q dtype tag, kv dtype tag)
}


def get_dtype_str(dtype: torch.dtype) -> str:
    if dtype == torch.uint8:
        return "nvfp4"
    if dtype == e4m3_dtype:
        return "fp8"
    if dtype in (torch.bfloat16, torch.float16):
        return "bf16"
    raise ValueError(f"No unified attention config tag for dtype: {dtype}")


def _axis_of(component: str) -> str:
    """``D_LEQ_128`` -> ``D``, ``DT_fp8_fp8`` -> ``DT``, ``SW`` -> ``SW``."""
    return "DT" if component.startswith("DT_") else component.split("_")[0]


def _canonical(key: str, axes: tuple) -> tuple:
    """Expand a key to one slot per axis, ``any`` where it says nothing."""
    slot = dict.fromkeys(axes, "any")
    if key != "any":
        for part in key.split(_SEP):
            slot[_axis_of(part)] = part
    return tuple(slot[a] for a in axes)


@functools.lru_cache(maxsize=256)
def _index(keys: tuple, axes: tuple) -> tuple:
    """Build ``(slots -> key, components used per axis)``.

    Cached on the key names, which is all it depends on.
    """
    parts = {a: set() for a in axes}
    for key in keys:
        if key != "any":
            for part in key.split(_SEP):
                parts[_axis_of(part)].add(part)
    return {_canonical(k, axes): k for k in keys}, parts


def _bound(component: str) -> int:
    return int(component.rsplit("_", 1)[1])


def _candidates(axis: str, value, parts: set) -> list[str]:
    """Components of `axis` matching `value`, most specific first."""
    kind = _AXIS_KIND[axis]
    if kind == "bool":
        return ([axis] if value else []) + ["any"]
    if kind == "enum":
        q, kv = value
        return [f"{axis}_{q}_{kv}", f"{axis}_{q}_any", f"{axis}_any_{kv}", "any"]

    leq = sorted((c for c in parts if c.startswith(f"{axis}_LEQ_")), key=_bound)
    geq = sorted(
        (c for c in parts if c.startswith(f"{axis}_GEQ_")), key=_bound, reverse=True
    )
    return (
        [c for c in leq if value <= _bound(c)]
        + [c for c in geq if value >= _bound(c)]
        + ["any"]
    )


def _lookup(table: dict, axes: tuple, values: dict) -> tuple:
    """Return ``(key, config)`` for this call."""
    if not axes:
        return None, dict(table)
    index, parts = _index(tuple(table), axes)
    per_axis = [_candidates(a, values[a], parts[a]) for a in axes]
    for slots in itertools.product(*per_axis):
        if slots in index:
            key = index[slots]
            return key, dict(table[key])
    raise KeyError(
        "no entry for "
        + " ".join(f"{a}={values[a]!r}" for a in axes)
        + f"; every table needs an 'any' entry (keys: {sorted(table)[:8]})"
    )


def compute_tile_params(config: dict, block_size: int) -> dict:
    """Derive TILE_SIZE from the tuned bounds and the runtime page size.
    """
    hi = config.pop("TILE_SIZE_MAX", None)
    lo = config.pop("TILE_SIZE_MIN", 1)
    if hi is not None:
        config["TILE_SIZE"] = max(lo, min(hi, triton.next_power_of_2(block_size)))
    return config


def _axis_values(
    head_size,
    max_seqlen_q,
    max_seqlen_k,
    sliding_window,
    shuffled_kv_cache,
    block_size,
    q_dtype,
    kv_dtype,
) -> dict:
    return {
        "D": head_size,
        "Q": max_seqlen_q,
        "BS": block_size,
        "SW": sliding_window > 0,
        "SKLT": max_seqlen_k < 2048,
        "SHUF": shuffled_kv_cache,
        "DT": (get_dtype_str(q_dtype), get_dtype_str(kv_dtype)),
    }


def _load(op: str, head_size, num_queries_per_kv, backend, arch) -> tuple:
    """Return ``(table, axes, is_tuned, cfg_dir)`` for one op."""
    cfg_dir = resolve_config_dir("attention", _CONFIG_NAME, backend=backend, arch=arch)
    config = load_config_json(f"{cfg_dir}/DEFAULT.json", required=False)
    if config is None:
        raise AssertionError(
            f"Required config file doesn't exist: {cfg_dir}/DEFAULT.json"
        )

    is_tuned = False
    for suffix in (f"D={head_size}-QPKV={num_queries_per_kv}", f"D={head_size}"):
        tuned = load_config_json(
            f"{cfg_dir}/{_CONFIG_NAME}-{suffix}.json", required=False
        )
        if tuned is not None:
            # Override per section, so a file tuning only attn_2d keeps the
            # default's other sections.
            config = {
                **config,
                "schema": {**config.get("schema", {}), **tuned.get("schema", {})},
                **{s: tuned[s] for s in _OPS if s in tuned},
            }
            is_tuned = True
            break

    if op not in config:
        raise KeyError(f"{_CONFIG_NAME}[{op}] in {cfg_dir}: file has no {op} section")
    axes = tuple(config.get("schema", {}).get(op, []))
    unknown = [a for a in axes if a not in _AXIS_KIND]
    assert not unknown, (
        f"{_CONFIG_NAME}[{op}] in {cfg_dir}: schema names unknown axes {unknown} "
        f"(known: {sorted(_AXIS_KIND)})"
    )
    return config[op], axes, is_tuned, cfg_dir


@functools.lru_cache(maxsize=1024 if USE_LRU_CACHE else 0)
def _get_unified_attention_config_cached(
    op: str,
    head_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sliding_window: int,
    shuffled_kv_cache: bool,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    num_queries_per_kv: int,
    block_size: int,
    backend: str,
    arch: str | None,
) -> tuple[dict, bool]:
    assert op in _OPS, f"Unknown config op {op!r}, expected one of {_OPS}"
    assert head_size > 0, "head_size must be positive"
    assert block_size > 0, "block_size must be positive"
    assert num_queries_per_kv > 0, "num_queries_per_kv must be positive"

    table, axes, is_tuned, _ = _load(op, head_size, num_queries_per_kv, backend, arch)
    values = _axis_values(
        head_size,
        max_seqlen_q,
        max_seqlen_k,
        sliding_window,
        shuffled_kv_cache,
        block_size,
        q_dtype,
        kv_dtype,
    )
    _, config = _lookup(table, axes, values)
    return compute_tile_params(config, block_size), is_tuned


def get_unified_attention_config(
    op: str,
    params,
    backend: str = "triton",
    arch: str | None = None,
) -> tuple[dict, bool]:
    """Load the config for one op.

    Args:
        op: ``attn_2d`` | ``attn_3d`` | ``reduce`` | ``kv_split``.
        params: the ``_UAParams`` for this call; every axis is read from it.
        backend: ``"triton"`` or ``"gluon"``. The two take disjoint config
            params, so a config from the wrong backend is not usable.
        arch: resolve another arch's table, for tooling and tests. Leave None
            in kernels.

    Returns:
        ``(config, is_tuned)``. The config is a fresh deep copy, safe to
        mutate. ``is_tuned`` is True only when a specialized ``-D=`` /
        ``-QPKV=`` file was hit; do not discard it.
    """
    config, is_tuned = _get_unified_attention_config_cached(
        op,
        params.head_size,
        params.max_seqlen_q,
        params.max_seqlen_k,
        params.sliding_window,
        params.shuffled_kv_cache,
        params.q_dtype,
        params.kv_cache_dtype,
        params.num_queries_per_kv,
        params.block_size,
        backend,
        arch,
    )
    return copy.deepcopy(config), is_tuned


def explain(op: str, params, backend: str = "triton", arch: str | None = None) -> str:
    """Report which entry a lookup lands on, and the config it yields."""
    table, axes, _, cfg_dir = _load(
        op, params.head_size, params.num_queries_per_kv, backend, arch
    )
    values = _axis_values(
        params.head_size,
        params.max_seqlen_q,
        params.max_seqlen_k,
        params.sliding_window,
        params.shuffled_kv_cache,
        params.block_size,
        params.q_dtype,
        params.kv_cache_dtype,
    )
    key, config = _lookup(table, axes, values)
    derived = compute_tile_params(dict(config), params.block_size)

    lines = [
        f"{_CONFIG_NAME}[{op}]  {cfg_dir}",
        f"  axes {list(axes)}",
        "  " + "  ".join(f"{a}={values[a]!r}" for a in axes),
        f"  matched: {key or '(flat config, no axes)'}",
        "  leaf:",
    ]
    lines += [f"    {k:22} = {v}" for k, v in sorted(config.items())]
    if "TILE_SIZE" in derived:
        lines.append(f"    {'TILE_SIZE':22} = {derived['TILE_SIZE']}   (derived)")
    return "\n".join(lines)
