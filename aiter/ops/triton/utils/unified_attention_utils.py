# SPDX-License-Identifier: MIT
# Copyright (C) 2026-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unified-attention config loading.

One entry per case, and **every entry is a complete config**. Reading the
config for a case is reading one line -- values are never assembled from
several places::

    {
      "schema": {"attn_2d": ["D", "Q", "SW", "DT"], "kv_split": ["DT", "BS"]},

      "attn_2d": {
        "D_LEQ_128.Q_LEQ_1.SW":   {...},   # small head, decode, sliding window
        "D_LEQ_128.Q_LEQ_1":      {...},   # small head, decode
        "D_LEQ_128.Q_GEQ_256":    {...},   # small head, long prefill
        "D_GEQ_512":              {...},   # large head
        "Q_LEQ_1":                {...},   # decode, any head size
        "any":                    {...}    # everything else
      },

      "reduce": {"num_warps": 2, "num_stages": 1, "waves_per_eu": 2}
    }

Each key component names its axis, so an axis that does not matter for a case
is simply left out -- there is no "any" padding, and a key reads as the
situation it covers. A section with no axes at all is a bare config, as
``reduce`` is above.

``schema`` lists the axes each section keys on -- only the ones this arch
actually uses, so the header is an honest summary. Bounds are not declared; the
keys state them, so adding a bucket is a one-line edit.

Resolution walks the cross product of each axis's candidates in schema order --
X_LEQ ascending, then X_GEQ descending, then the looser dtype fallbacks, then
"any". The enumeration order *is* the priority: leftmost axis most significant,
which is what makes head_size outrank max_seqlen_q as the old if-else nesting
did. Keys are expanded to one slot per axis once at load time, so no ranking
over variable-length keys is needed. Entries are stored most specific first, so
the file reads in the order the lookup tries them.

Component grammar per axis kind:
  numeric  ``D_LEQ_128`` / ``D_GEQ_512``
  boolean  ``SW`` -- present means the flag is set; absent means don't care
  enum     ``DT_fp8_fp8`` -- the (q, kv) dtype tag pair, with ``DT_fp8_any`` /
           ``DT_any_fp8`` as progressively looser fallbacks

Separator is ``.`` rather than ``_``: axis names collide otherwise, since D is
a prefix of DT and ``DT_any`` would read as axis D.
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
# Separates key components. Not "_": axis names collide under it, since D
# is a prefix of DT and "DT_any" would read as axis D.
_SEP = "."
_OPS = ("attn_2d", "attn_3d", "reduce", "kv_split")

# How each axis name is bucketed. Adding an axis here plus a value in
# _axis_values() is all the Python a new axis needs.
_AXIS_KIND = {
    "D": "num",  # head_size
    "Q": "num",  # max_seqlen_q -- Q_LEQ_1 is decode
    "BS": "num",  # block_size, i.e. the KV page size
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
    """D_LEQ_128 -> D;  DT_fp8_fp8 -> DT;  SW -> SW."""
    return "DT" if component.startswith("DT_") else component.split("_")[0]


def _canonical(key: str, axes: tuple) -> tuple:
    """A key with one slot per axis, "any" where the key says nothing.

    Keys name only the axes that matter, so they vary in length; expanding them
    to a fixed shape once at load time is what lets the lookup be a plain
    cross-product walk instead of a specificity ranking over variable keys.
    """
    slot = dict.fromkeys(axes, "any")
    if key != "any":
        for part in key.split(_SEP):
            slot[_axis_of(part)] = part
    return tuple(slot[a] for a in axes)


@functools.lru_cache(maxsize=256)
def _index(keys: tuple, axes: tuple) -> tuple:
    """(canonical slots -> original key, components present per axis).

    Keyed on the table's key names alone -- they are what the index depends on,
    and unlike the configs they are hashable, so this is computed once per
    table rather than once per shape.
    """
    idx = {_canonical(k, axes): k for k in keys}
    seen: dict = {a: set() for a in axes}
    for key in keys:
        if key != "any":
            for part in key.split(_SEP):
                seen[_axis_of(part)].add(part)
    return idx, seen


def _candidates(axis: str, value, present: set) -> list[str]:
    """Components to try for this axis, most specific first -- X_LEQ ascending,
    then X_GEQ descending, then the looser dtype fallbacks, then "any".

    Bounds come from the components the table actually uses, so adding a bucket
    is a one-line edit with nothing to declare elsewhere.
    """
    kind = _AXIS_KIND[axis]
    if kind == "num":

        def bound(component):
            return int(component.rsplit("_", 1)[1])

        leq = sorted((c for c in present if c.startswith(f"{axis}_LEQ_")), key=bound)
        geq = sorted(
            (c for c in present if c.startswith(f"{axis}_GEQ_")),
            key=bound,
            reverse=True,
        )
        out = [c for c in leq if value <= bound(c)]
        out += [c for c in geq if value >= bound(c)]
        return out + ["any"]
    if kind == "bool":
        return ([axis] if value else []) + ["any"]
    q, k = value
    return [f"{axis}_{q}_{k}", f"{axis}_{q}_any", f"{axis}_any_{k}", "any"]


def _resolve(table: dict, axes: list, values: dict) -> dict:
    """Find the entry for this call.

    Walks the cross product of each axis's candidates in schema order, so the
    enumeration order *is* the priority -- leftmost axis most significant, most
    specific component first. No ranking function, and no failure path in a
    well-formed table, which always has an "any".
    """
    if not axes:
        return dict(table)
    axes = tuple(axes)
    idx, seen = _index(tuple(table), axes)
    per_axis = [_candidates(a, values[a], seen[a]) for a in axes]
    for combo in itertools.product(*per_axis):
        if combo in idx:
            return dict(table[idx[combo]])
    raise KeyError(
        "no entry for "
        + " ".join(f"{a}={values[a]!r}" for a in axes)
        + f"; a table needs an 'any' entry. Keys: {sorted(table)[:8]}"
    )


def compute_tile_params(config: dict, block_size: int) -> dict:
    """TILE_SIZE from the tuned bounds and the runtime page size.

    The JSON holds the bounds, the runtime picks the value -- the same split
    GEMM uses in ``compute_splitk_params``, where the tuned ``NUM_KSPLIT`` meets
    the runtime ``K``. Two consequences worth keeping:

    * the whole ``BS_LEQ_*`` ladder disappears from the attention entries, and
    * the lookup becomes *total* over page sizes. A bucket ladder only covers up
      to its largest named bound and from its smallest, so a 48-token page fell
      in the gap and raised KeyError; ``max(lo, min(hi, pow2))`` is defined for
      every positive page size.

    There is no fp8 special case here. Where fp8 needs a different tile it gets
    an ``fp8_fp8`` bucket like any other difference, so the leaf states what fp8
    runs instead of a separate key applying a floor from a leaf that never
    mentions fp8.
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
    q_tag,
    kv_tag,
) -> dict:
    """The runtime value of every axis the schema may name."""
    return {
        "D": head_size,
        "Q": max_seqlen_q,
        "BS": block_size,
        "SW": sliding_window > 0,
        "SKLT": max_seqlen_k < 2048,
        "SHUF": shuffled_kv_cache,
        "DT": (q_tag, kv_tag),
    }


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
    """Internal, cached. Use ``get_unified_attention_config()``, which
    deep-copies so callers can mutate the result freely."""
    assert op in _OPS, f"Unknown config op {op!r}, expected one of {_OPS}"
    assert head_size > 0, "head_size must be positive"
    assert block_size > 0, "block_size must be positive"
    assert num_queries_per_kv > 0, "num_queries_per_kv must be positive"

    cfg_dir = resolve_config_dir("attention", _CONFIG_NAME, backend=backend, arch=arch)
    config_dict = load_config_json(f"{cfg_dir}/DEFAULT.json", required=False)
    if config_dict is None:
        raise AssertionError(
            f"Required config file doesn't exist: {cfg_dir}/DEFAULT.json"
        )

    is_tuned = False
    for suffix in (f"D={head_size}-QPKV={num_queries_per_kv}", f"D={head_size}"):
        specialized = load_config_json(
            f"{cfg_dir}/{_CONFIG_NAME}-{suffix}.json", required=False
        )
        if specialized is not None:
            # Override section by section, so a file that tunes only attn_2d
            # keeps the default's attn_3d / reduce / kv_split rather than
            # blanking them.
            merged = dict(config_dict)
            merged["schema"] = {
                **config_dict.get("schema", {}),
                **specialized.get("schema", {}),
            }
            for section in _OPS:
                if section in specialized:
                    merged[section] = specialized[section]
            config_dict, is_tuned = merged, True
            break

    where = f"{_CONFIG_NAME}[{op}] in {cfg_dir}"
    if op not in config_dict:
        raise KeyError(f"{where}: file has no {op} section")
    axes = config_dict.get("schema", {}).get(op, [])
    for axis in axes:
        assert axis in _AXIS_KIND, (
            f"{where}: schema declares axis {axis!r}, which the resolver does "
            f"not know how to bucket (known: {sorted(_AXIS_KIND)})"
        )

    q_tag, kv_tag = get_dtype_str(q_dtype), get_dtype_str(kv_dtype)
    values = _axis_values(
        head_size,
        max_seqlen_q,
        max_seqlen_k,
        sliding_window,
        shuffled_kv_cache,
        block_size,
        q_tag,
        kv_tag,
    )

    config = _resolve(config_dict[op], axes, values)
    config = compute_tile_params(config, block_size)
    return config, is_tuned


def get_unified_attention_config(
    op: str,
    params,
    backend: str = "triton",
    arch: str | None = None,
) -> tuple[dict, bool]:
    """Load a unified attention config for one op.

    Args:
        op: ``attn_2d`` | ``attn_3d`` | ``reduce`` | ``kv_split``.
        params: the ``_UAParams`` for this call. Every axis is read from it, so
            callers no longer build a composite key, and adding an axis to a
            config file needs no change at the call site.
        backend: ``"triton"`` or ``"gluon"`` -- the two take disjoint config
            params, so a config from the wrong backend is not usable.
        arch: overrides the running architecture. For tooling and tests that
            resolve another arch's table; leave None in kernels.

    Returns:
        ``(config, is_tuned)`` -- a fresh deep copy, safe to mutate, and
        ``is_tuned`` True only when a specialized ``-D=`` / ``-QPKV=`` file was
        hit. Do not discard ``is_tuned``.
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
    """Trace which buckets a lookup walked and which leaf it landed on.

    Answers "what config runs for this situation" for the actual runtime
    inputs, instead of asking the reader to reconstruct a lookup key by hand.
    """
    cfg_dir = resolve_config_dir("attention", _CONFIG_NAME, backend=backend, arch=arch)
    config_dict = load_config_json(f"{cfg_dir}/DEFAULT.json")
    axes = config_dict.get("schema", {}).get(op, [])
    q_tag = get_dtype_str(params.q_dtype)
    kv_tag = get_dtype_str(params.kv_cache_dtype)
    values = _axis_values(
        params.head_size,
        params.max_seqlen_q,
        params.max_seqlen_k,
        params.sliding_window,
        params.shuffled_kv_cache,
        params.block_size,
        q_tag,
        kv_tag,
    )

    axes = tuple(axes)
    if not axes:
        leaf, key = dict(config_dict[op]), "(flat config, no axes)"
    else:
        table = config_dict[op]
        idx, seen = _index(tuple(table), axes)
        per_axis = [_candidates(a, values[a], seen[a]) for a in axes]
        leaf, key = None, None
        for combo in itertools.product(*per_axis):
            if combo in idx:
                key = idx[combo]
                leaf = dict(table[key])
                break
    lines = [f"{_CONFIG_NAME}[{op}]  {cfg_dir}", f"  axes {list(axes)}"]
    lines.append("  " + "  ".join(f"{a}={values[a]!r}" for a in axes))
    if leaf is None:
        lines.append("  UNRESOLVED")
        return "\n".join(lines)
    lines.append(f"  matched: {key}")
    resolved = compute_tile_params(dict(leaf), params.block_size)
    lines.append("  leaf:")
    for key in sorted(leaf):
        lines.append(f"    {key:22} = {leaf[key]}")
    if "TILE_SIZE" in resolved:
        lines.append(f"    {'TILE_SIZE':22} = {resolved['TILE_SIZE']}   (derived)")
    return "\n".join(lines)
