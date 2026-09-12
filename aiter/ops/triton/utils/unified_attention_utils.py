# SPDX-License-Identifier: MIT
# Copyright (C) 2026-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Loads tuning configs for unified attention.

Each op is a flat table of configs, keyed by the case it covers. A key lists
only the axes that matter, joined by '.':

    {
      "schema": {"attn_2d": ["D", "Q", "SW", "DT"]},

      "attn_2d": {
        "D_LEQ_128.Q_LEQ_1.SW": {...},   # head_size <= 128, decode, sliding window
        "D_LEQ_128.Q_LEQ_1":    {...},   # head_size <= 128, decode
        "D_GEQ_512":            {...},   # head_size >= 512, any query length
        "DT_fp8_fp8":           {...},   # fp8 query and KV cache
        "any":                  {...}
      },

      "reduce": {"num_warps": 2, "num_stages": 1, "waves_per_eu": 2}
    }

Lookup walks the axes in the order "schema" lists them and takes the first key
that exists: LEQ bounds ascending, then GEQ descending, then "any". So the
leftmost axis wins: D before Q is what makes head_size outrank max_seqlen_q.
Dtypes fall back too: DT_fp8_fp8, then DT_fp8_any, DT_any_fp8, then "any".

Because the leftmost axis wins outright, D_GEQ_256 beats D_GEQ_129.Q_GEQ_256 at
head_size 256 despite saying less. An entry narrowing a rightward axis has to be
repeated at every bound of a leftward one, which is why some keys differ only in
that.

A section with no axes, like reduce above, is just a config.

Tile size and number of splits (segments) are derived from the following parameters:
TILE_SIZE_MIN/MAX, and MIN_SEGMENTS/MAX_SEGMENTS/SEGMENTS_PER_CU.

An "<op>.OPT_IN" section holds entries that exist only when the user opts in
(_load); a LARGE_KV block inside one applies only above a KV cache size
(_resolve_large_kv).
"""

import copy
import dataclasses
import functools
import itertools
import os

import torch
import triton

from aiter.ops.triton.utils.config_utils import (
    USE_LRU_CACHE,
    load_config_json,
    resolve_config_dir,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.types import e4m3_dtype

_LOGGER = AiterTritonLogger()

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


# Opt-in section suffix, and the env var that admits it. Unset, the table loads
# exactly as written. These configs trade occupancy for registers and are tuned
# against one shape family, so whether that suits a deployment is its own call.
_OPT_IN = ".OPT_IN"

_USE_OPT_IN = os.environ.get("AITER_TRITON_UA_FAST_PATH", "") == "1"

_LARGE_KV = "LARGE_KV"

# Below this the backend indexes the cache with 32-bit buffer ops, and against
# that different code the LARGE_KV overrides invert: gfx950 d=256 prefill runs
# 6.2 -> 4.7ms above the limit, 5.2 -> 14.9ms below. The crossover is exact.
_BUFFER_OP_LIMIT = 2**31

# Compiler options only newer Triton takes, so a LARGE_KV block naming one is
# skipped where the backend has no such option.
_OPTIONAL_COMPILER_OPTIONS = frozenset(("llvm_fn_attrs", "schedule_hint"))


@functools.lru_cache(maxsize=1)
def _compiler_option_names() -> frozenset:
    """Compiler options the installed Triton takes as launch kwargs, if askable."""
    try:
        from triton.compiler.compiler import make_backend
        from triton.runtime.driver import driver

        backend = make_backend(driver.active.get_current_target())
        return frozenset(f.name for f in dataclasses.fields(backend.parse_options({})))
    except (ImportError, AttributeError, TypeError, RuntimeError):
        # older Triton, unfamiliar backend, or no GPU: decline rather than risk
        # a launch the backend rejects
        return frozenset()


def _resolve_large_kv(config: dict, params) -> dict:
    """Apply an entry's LARGE_KV overrides, if the cache is large enough for them.

    They lose about threefold below _BUFFER_OP_LIMIT, and cost bf16 sixfold at
    any size, so only DT_fp8_fp8 entries carry them. Counting elements rather
    than bytes leaves a borderline cache on the safe side of the limit.
    """
    large_kv = config.pop(_LARGE_KV, None)
    if large_kv is None:
        return config

    unsupported = sorted(
        _OPTIONAL_COMPILER_OPTIONS.intersection(large_kv) - _compiler_option_names()
    )
    if unsupported:
        _LOGGER.warning(
            "unified attention: this Triton takes no %s, so the opt-in entry's "
            "large-KV overrides are unavailable",
            ", ".join(unsupported),
        )
        return config

    if params.k.numel() < _BUFFER_OP_LIMIT:
        _LOGGER.debug(
            "unified attention: KV cache of %d elements is under the %d the "
            "large-KV overrides need",
            params.k.numel(),
            _BUFFER_OP_LIMIT,
        )
        return config

    config.update(large_kv)
    return config


def compute_tile_params(config: dict, block_size: int) -> dict:
    """Derive TILE_SIZE from the tuned bounds and the runtime page size."""
    if "TILE_SIZE_MIN" not in config and "TILE_SIZE_MAX" not in config:
        return config
    hi = config.pop("TILE_SIZE_MAX", None)
    lo = config.pop("TILE_SIZE_MIN", 1)
    tile = triton.next_power_of_2(block_size)
    config["TILE_SIZE"] = max(lo, tile if hi is None else min(hi, tile))
    return config


def compute_segment_params(config: dict, params) -> dict:
    """Derive NUM_SEGMENTS: how many ways to split the KV range for one query.

    The reduce section carries the same parameters plus SMALL_SPLIT_MAX, and
    gets num_warps out of this instead of a segment count: one warp is enough
    when the split landed on its floor.
    """
    if "SEGMENTS_PER_CU" not in config:
        return config
    small_split_max = config.pop("SMALL_SPLIT_MAX", None)
    per_cu = config.pop("SEGMENTS_PER_CU")
    lo = config.pop("MIN_SEGMENTS", 1)
    cap = config.pop("MAX_SEGMENTS", None)
    tile_lo = config.pop("SEGMENT_TILE_MIN", 1)
    tile_hi = config.pop("SEGMENT_TILE_MAX", None)

    # tokens one segment must cover, so the split never outruns the context
    tile = triton.next_power_of_2(params.block_size)
    tile = max(tile_lo, tile if tile_hi is None else min(tile_hi, tile))
    limit = triton.cdiv(params.max_seqlen_k, tile)
    if cap is not None:
        limit = min(cap, limit)

    budget = params.num_sms * per_cu
    prgms = max(1, params.num_2d_prgms)
    share = triton.cdiv(budget, prgms)
    segments = triton.next_power_of_2(max(min(lo, limit), min(limit, max(1, share))))
    if small_split_max is None:
        config["NUM_SEGMENTS"] = segments
    elif segments <= min(small_split_max, limit):
        # the split landed on its floor: too few segments to be worth 2 warps
        config["num_warps"] = 1
    return config


def _derive(config: dict, params) -> dict:
    """Fill in the parts of a config that depend on the launch, not the key.

    Only the trailing compute_tile_params call can be a repeat: it re-derives
    TILE_SIZE if the large-KV overrides moved its bounds, and is a no-op if not.
    """
    config = compute_segment_params(config, params)
    return compute_tile_params(_resolve_large_kv(config, params), params.block_size)


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


def _load(op: str, backend, arch) -> tuple:
    """Return ``(table, axes, cfg_dir)`` for one op."""
    cfg_dir = resolve_config_dir("attention", _CONFIG_NAME, backend=backend, arch=arch)
    config = load_config_json(f"{cfg_dir}/DEFAULT.json", required=False)
    if config is None:
        raise AssertionError(
            f"Required config file doesn't exist: {cfg_dir}/DEFAULT.json"
        )

    if op not in config:
        raise KeyError(f"{_CONFIG_NAME}[{op}] in {cfg_dir}: file has no {op} section")
    axes = tuple(config.get("schema", {}).get(op, []))
    unknown = [a for a in axes if a not in _AXIS_KIND]
    assert not unknown, (
        f"{_CONFIG_NAME}[{op}] in {cfg_dir}: schema names unknown axes {unknown} "
        f"(known: {sorted(_AXIS_KIND)})"
    )

    table = config[op]
    if _USE_OPT_IN:
        # A merge, not an edit: an opt-in key may be new to the table, and then
        # has to enter the lookup as a candidate rather than replace a config.
        table = {**table, **config.get(op + _OPT_IN, {})}
    return table, axes, cfg_dir


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
    block_size: int,
    backend: str,
    arch: str | None,
) -> dict:
    assert op in _OPS, f"Unknown config op {op!r}, expected one of {_OPS}"
    assert head_size > 0, "head_size must be positive"
    assert block_size > 0, "block_size must be positive"

    table, axes, _ = _load(op, backend, arch)
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
    return compute_tile_params(config, block_size)


def get_unified_attention_config(
    op: str,
    params,
    backend: str = "triton",
    arch: str | None = None,
) -> dict:
    """Load the config for one op.

    Args:
        op: attn_2d | attn_3d | reduce | kv_split.
        params: the _UAParams for this call; every axis is read from it.
        backend: "triton" or "gluon". The two take disjoint config
            params, so a config from the wrong backend is not usable.
        arch: resolve another arch's table, for tooling and tests. Leave None
            in kernels.

    Returns:
        The config, as a fresh deep copy that is safe to mutate.
    """
    config = _get_unified_attention_config_cached(
        op,
        params.head_size,
        params.max_seqlen_q,
        params.max_seqlen_k,
        params.sliding_window,
        params.shuffled_kv_cache,
        params.q_dtype,
        params.kv_cache_dtype,
        params.block_size,
        backend,
        arch,
    )
    # Derived out here, not in the cached lookup: these read the launch's
    # program count and KV cache, neither of which is in the cache key.
    return _derive(copy.deepcopy(config), params)


def explain(op: str, params, backend: str = "triton", arch: str | None = None) -> str:
    """Report which entry a lookup lands on, and the config it yields."""
    table, axes, cfg_dir = _load(op, backend, arch)
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
    derived = _derive(compute_tile_params(dict(config), params.block_size), params)

    lines = [
        f"{_CONFIG_NAME}[{op}]  {cfg_dir}",
        f"  axes {list(axes)}",
        "  " + "  ".join(f"{a}={values[a]!r}" for a in axes),
        f"  matched: {key or '(flat config, no axes)'}",
        "  leaf:",
    ]
    lines += [f"    {k:22} = {v}" for k, v in sorted(config.items())]
    for name, value in sorted(derived.items()):
        if config.get(name) != value:
            lines.append(f"    {name:22} = {value}   (derived)")
    return "\n".join(lines)
