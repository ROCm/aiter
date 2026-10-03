# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The brute-force search space and the pruning rules.

SEARCH_SPACE lists the candidate values per config key. The sweep always runs the full
product of these lists for the keys found in the kernel family's DEFAULT.json; there are no
command-line overrides. Edit this table to change what is swept. A single-value list pins
that key. A DEFAULT.json key that is missing here stops the program: add it here first.

build_space() is the planning flow: load the defaults -> pick candidate values per key
-> generate the combinations -> drop the ones a rule rejects. Every should_skip_* / exceeds_*
function returns True to reject a config.
"""

import itertools
import math

from _utils import cdiv, next_pow2

SEARCH_SPACE = {
    "BLOCK_SIZE_M": [4, 8, 16, 32, 64, 128, 256, 512, 1024],
    "BLOCK_SIZE_N": [4, 8, 16, 32, 64, 128, 256, 512, 1024],
    "BLOCK_SIZE_K": [4, 8, 16, 32, 64, 128, 256, 512, 1024],
    "GROUP_SIZE_M": [1, 4, 8, 16],
    "num_warps": [1, 2, 4, 8],
    "num_stages": [1, 2],
    "waves_per_eu": [1, 2, 4, 6, 8],
    "matrix_instr_nonkdim": [16],
    "cache_modifier": [".cg", None],
    "NUM_KSPLIT": [1, 3, 4, 7, 8, 14, 16, 28],
    "NUM_BUFFERS": [1, 2, 3, 4, 6, 8],
    "kernel_type": ["bandwidth_bound", "compute_bound"],
    "num_ctas": [1, 2, 4, 8],
    "CTAS_M": [1, 2, 4],
    "CTAS_N": [1, 2, 4],
    "LOOP_UNROLL_FACTOR": [1, 2],
    "B_SCALE_TDM": [True, False],
    "kpack": [1, 2],
    "persistent": [False],
}

# Families that spell a key differently share its table row and shape rules;
# the configs keep the family's own spelling.
ALIASES = {
    "BLOCK_M": "BLOCK_SIZE_M",
    "BLOCK_N": "BLOCK_SIZE_N",
    "BLOCK_K": "BLOCK_SIZE_K",
}

# The gluon compiler ignores these launch options, so under gluon they stay at the default.
TRITON_ONLY_KEYS = {"matrix_instr_nonkdim", "kpack"}


class UnknownConfigKey(Exception):
    pass


def load_defaults(spec, backend):
    """Keys and buckets of the family's DEFAULT.json -> (keys, buckets, default_path).

    The keys are the union over all buckets, in file order. M_BOUNDS, _note and friends are
    not buckets.
    """
    from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir

    config_dir = resolve_config_dir("gemm", spec.config_name, backend=backend)
    default_path = f"{config_dir}/DEFAULT.json"
    table = load_config_json(default_path, required=False)
    if table is None:
        raise FileNotFoundError(f"Required config file doesn't exist: {default_path}")
    buckets = {name: entry for name, entry in table.items() if isinstance(entry, dict)}
    keys = list(dict.fromkeys(key for entry in buckets.values() for key in entry))
    return keys, buckets, default_path


def default_values(spec, backend, M, keys, buckets):
    """The DEFAULT.json bucket the loader picks for M, filled up with keys only other buckets have."""
    from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

    values, _ = get_gemm_config(
        spec.config_name, M, bounds=spec.bounds, backend=backend
    )
    for key in keys:
        if key not in values:
            values[key] = next(entry[key] for entry in buckets.values() if key in entry)
    return values


def candidate_values(spec, backend, keys, defaults, default_path):
    """Candidate list per key. A one-element list means the key is pinned (not swept)."""
    values, pinned = {}, []
    for key in keys:
        name = ALIASES.get(key, key)
        if name not in SEARCH_SPACE:
            raise UnknownConfigKey(
                f"Unknown config key '{key}' in {default_path}: "
                "add it to SEARCH_SPACE in space.py first"
            )
        candidates = SEARCH_SPACE[name]
        if backend == "gluon" and key in spec.gluon_candidates:
            candidates = spec.gluon_candidates[key]
        gluon_never_reads = backend == "gluon" and (
            key in spec.gluon_ignored_keys or name in TRITON_ONLY_KEYS
        )
        if gluon_never_reads:
            candidates = [
                defaults[key]
            ]  # sweeping it would only repeat the same kernel
        values[key] = list(candidates)
        if len(values[key]) == 1:
            pinned.append(key)
    return values, pinned


def filter_by_shape(key, values, shape, backend, allow_oversized_tiles):
    """Drop values the shape makes pointless; always keep at least one.

    Block sizes above the next power of two of their dimension are dropped unless
    allow_oversized_tiles; NUM_KSPLIT values that do not divide K are always dropped.
    """
    name = ALIASES.get(key, key)
    M, N, K = shape["M"], shape["N"], shape["K"]
    if name == "NUM_KSPLIT":
        kept = [v for v in values if K % v == 0]
    elif name in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"):
        if name == "BLOCK_SIZE_M" and backend == "gluon":
            values = [
                v for v in values if v >= 16
            ] or values  # gluon tiles start at 16 rows
        dim = {"BLOCK_SIZE_M": M, "BLOCK_SIZE_N": N, "BLOCK_SIZE_K": K}[name]
        kept = (
            values
            if allow_oversized_tiles
            else [v for v in values if v <= next_pow2(dim)]
        )
    else:
        return values
    return kept or [min(values)]


def should_skip_generic(shape, config, backend):
    """Pruning rules shared by all GEMM kernels (the old pre-pruning rules). True = reject."""
    M, K = shape["M"], shape["K"]
    block_m = config.get("BLOCK_SIZE_M", config.get("BLOCK_M"))
    block_k = config.get("BLOCK_SIZE_K", config.get("BLOCK_K"))
    split_k = config.get("NUM_KSPLIT", 1)
    group_size_m = config.get("GROUP_SIZE_M", 1)
    num_stages = config.get("num_stages")
    k_per_split = K // split_k
    if split_k > 1 and group_size_m > 1:
        return True  # tile-order swizzling is not combined with split-K
    if group_size_m > 1 and block_m is not None and cdiv(M, block_m) == 1:
        return True  # a single M tile: swizzling is the identity, same kernel as GROUP_SIZE_M=1
    if block_k is None:
        return False
    if block_k >= 2 * k_per_split:
        return True  # the K tile is more than double the K each split works on
    if backend == "triton" and num_stages is not None:
        # triton pipelines (num_stages > 1) exactly when there are several K iterations
        return (num_stages > 1) != (block_k < k_per_split)
    return False


def exceeds_lds(config, bits, arch, backend):
    """True when buffers x (A tile + B tile) cannot fit the LDS; such a config never compiles."""
    from aiter.ops.triton.utils._triton.arch_info import _LDS_CAP_BYTES

    block_m = config.get("BLOCK_SIZE_M", config.get("BLOCK_M"))
    block_n = config.get("BLOCK_SIZE_N", config.get("BLOCK_N"))
    block_k = config.get("BLOCK_SIZE_K", config.get("BLOCK_K"))
    if None in (block_m, block_n, block_k):
        return False
    # gluon keeps NUM_BUFFERS (or num_stages, where that is what the key means) tile pairs
    # resident; the triton pipeliner keeps num_stages - 1 (lenient bound)
    num_stages = config.get("num_stages", 1)
    buffers = config.get(
        "NUM_BUFFERS", num_stages if backend == "gluon" else max(num_stages - 1, 1)
    )
    tile_bytes = (block_m * block_k * bits[0] + block_n * block_k * bits[1]) / 8
    return buffers * tile_bytes > _LDS_CAP_BYTES.get(arch, 64 * 1024)


def build_space(spec, shape, backend, arch, kernel_should_skip):
    """All configs to benchmark for one shape, plus a report of how the space was built."""
    keys, buckets, default_path = load_defaults(spec, backend)
    defaults = default_values(spec, backend, shape["M"], keys, buckets)
    values, pinned = candidate_values(spec, backend, keys, defaults, default_path)

    def shape_filtered(allow_oversized_tiles):
        return {
            k: (
                v
                if k in pinned
                else filter_by_shape(k, v, shape, backend, allow_oversized_tiles)
            )
            for k, v in values.items()
        }

    def prune(candidate_lists):
        configs = []
        skipped = {"generic rules": 0, "LDS": 0, "kernel rules": 0}
        for combination in itertools.product(*candidate_lists.values()):
            config = dict(zip(keys, combination))
            if should_skip_generic(shape, config, backend):
                skipped["generic rules"] += 1
                continue
            if exceeds_lds(config, spec.bits, arch, backend):
                skipped["LDS"] += 1
                continue
            if kernel_should_skip is not None and kernel_should_skip(config):
                skipped["kernel rules"] += 1
                continue
            configs.append(config)
        return configs, skipped

    allow_oversized_tiles = False
    shape_filtered_values = shape_filtered(allow_oversized_tiles)
    configs, skipped = prune(shape_filtered_values)
    if not configs and skipped["kernel rules"]:
        # the kernel rejects every tile that fits the shape (it needs 64-row tiles at M=16, say):
        # allow block sizes above the shape; NUM_KSPLIT must still divide K
        allow_oversized_tiles = True
        shape_filtered_values = shape_filtered(allow_oversized_tiles)
        configs, skipped = prune(shape_filtered_values)

    report = {
        "default_path": default_path,
        "keys": keys,
        "pinned": pinned,
        "candidate_values": shape_filtered_values,
        "oversized_tiles_allowed": allow_oversized_tiles,
        "raw": math.prod(len(v) for v in shape_filtered_values.values()),
        "skipped": skipped,
        "final": len(configs),
    }
    return configs, report
