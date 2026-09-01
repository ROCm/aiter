# SPDX-License-Identifier: MIT
# Copyright (C) 2026-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unified-attention config loading.

One nesting level per axis, and **every leaf is a complete config**. Reading
the config for a case is reading one leaf -- values are never assembled from
several places::

    {
      "schema": {"attn_2d": ["D", "Q", "SW", "DT"], "kv_split": ["D", "DT", "BS"]},
      "attn_2d": {
        "D_LEQ_128": {                                  # small head
          "Q_LEQ_1": {"BLOCK_M": 16, "num_warps": 2,    #   decode
                      "num_stages": 3, "waves_per_eu": 2,
                      "TILE_SIZE_MIN": 16, "TILE_SIZE_MAX": 64}
        },
        "D_GEQ_512": {                                  # large head
          "Q_LEQ_1": {...},                             #   decode
          "any":     {...}                              #   prefill
        },
        "any": {                                        # everything else
          "Q_LEQ_1":   {...},                           #   decode
          "Q_GEQ_256": {...},                           #   long prefill
          "any":       {...}                            #   prefill
        }
      }
    }

``schema`` lists the axes each section branches on -- only the ones this arch
actually uses, so the header is an honest summary of the file. Bounds are not
declared; the bucket keys state them. A section that is a single flat config
needs no entry at all. Nesting order *is* precedence: ``D`` outside ``Q`` is
what makes head_size outrank max_seqlen_q, as the old if-else nesting did.

Resolution is a plain descent in the GEMM search order -- ``X_LEQ`` ascending,
then ``X_GEQ`` descending, then ``any``. Every level carries an ``any``
(enforced at load time), so a lookup always lands and there is no backtracking
or failure path. A level may be omitted entirely where it never changes
anything, and each level's axis is read from its own bucket names rather than
from a declared order the file might not follow.

Bucket grammar per axis kind -- every bucket names its axis:
  numeric  ``D_LEQ_128`` / ``D_GEQ_512``
  boolean  ``SW`` -- present means the flag is set; ``any`` is the other case
  enum     ``DT=fp8_fp8`` -- the (q, kv) dtype tag pair, with ``DT=fp8_any`` /
           ``DT=any_fp8`` / ``any`` as progressively looser fallbacks
"""

import copy
import functools

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


def _axis_of(key: str) -> str:
    """Every bucket names its axis: D_LEQ_128, SW, DT=fp8_fp8."""
    return key.split("=")[0].split("_LEQ_")[0].split("_GEQ_")[0]


def _candidates(axis: str, value, node: dict) -> list[str]:
    """Bucket names to try for this axis, in resolution order -- X_LEQ
    ascending, then X_GEQ descending, then the looser fallbacks, then "any".

    Bounds are read from the node's own keys. Declaring them again in the schema
    was a second source of truth for a fact the keys already state, and it meant
    adding one bucket required editing two places.
    """
    kind = _AXIS_KIND[axis]
    if kind == "num":

        def bound(key):
            return int(key.rsplit("_", 1)[1])

        leq = sorted((k for k in node if k.startswith(f"{axis}_LEQ_")), key=bound)
        geq = sorted(
            (k for k in node if k.startswith(f"{axis}_GEQ_")), key=bound, reverse=True
        )
        out = [k for k in leq if value <= bound(k)]
        out += [k for k in geq if value >= bound(k)]
        return out + ["any"]
    if kind == "bool":
        return ([axis] if value else []) + ["any"]
    q, k = value
    return [f"{axis}={q}_{k}", f"{axis}={q}_any", f"{axis}=any_{k}", "any"]


def _is_leaf(node: dict) -> bool:
    """A leaf holds scalars; a level holds sub-dicts."""
    return not any(isinstance(v, dict) for v in node.values())


def _resolve(node: dict, values: dict) -> dict:
    """Descend to the leaf for this call.

    Each level's axis is read from its own bucket names, so the tree drives the
    descent rather than a declared order the file might not follow. Every level
    carries an "any" -- enforced by the config tests, not re-checked per lookup
    -- so in a well-formed table the descent always lands and there is no
    backtracking or failure path.
    """
    while not _is_leaf(node):
        axis = _axis_of(next(k for k in node if k != "any"))
        for key in _candidates(axis, values[axis], node):
            if key in node:
                node = node[key]
                break
        else:
            # Structure is checked in the config tests, not here -- this guard
            # exists so a malformed table fails loudly at the offending level
            # rather than looping forever on a node it cannot descend.
            raise KeyError(
                f"no bucket for {axis}={values[axis]!r} and no 'any' fallback "
                f"among {sorted(node)}"
            )
    return dict(node)


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

    config = _resolve(config_dict[op], values)
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

    trail: list[str] = []
    node = config_dict[op]
    while not _is_leaf(node):
        axis = _axis_of(next(k for k in node if k != "any"))
        for key in _candidates(axis, values[axis], node):
            if key in node:
                trail.append(f"{'  ' * len(trail)}{axis}={values[axis]!r} -> {key}")
                node = node[key]
                break
    leaf = dict(node)
    lines = [f"{_CONFIG_NAME}[{op}]  {cfg_dir}", f"  axes {list(axes)}", "  path:"]
    lines += [f"    {step}" for step in trail]
    resolved = compute_tile_params(dict(leaf), params.block_size)
    lines.append("  leaf:")
    for key in sorted(leaf):
        lines.append(f"    {key:22} = {leaf[key]}")
    if "TILE_SIZE" in resolved:
        lines.append(f"    {'TILE_SIZE':22} = {resolved['TILE_SIZE']}   (derived)")
    return "\n".join(lines)
