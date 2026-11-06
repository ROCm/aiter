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
        "D_EQ_192":             {...},   # head_size == 192 and nothing else
        "D_GEQ_512":            {...},   # head_size >= 512, any query length
        "DT_fp8_fp8":           {...},   # fp8 query and KV cache
        "any":                  {...}
      },

      "reduce": {"num_warps": 2, "num_stages": 1, "waves_per_eu": 2}
    }

Lookup walks the axes in the order "schema" lists them and takes the first key
that exists: an exact EQ first, then LEQ bounds ascending, then GEQ
descending, then "any". So the leftmost axis wins: D before Q is what makes
head_size outrank max_seqlen_q.
Dtypes fall back too: DT_fp8_fp8, then DT_fp8_any, DT_any_fp8, then "any".
SW is the keys a row can see, min(window, max_seqlen_k), and 0 with no window:
SW_GEQ_<n> targets a window size, a bare SW matches any window, and a table
may not mix SW_LEQ and SW_GEQ keys.

A section with no axes, like reduce above, is just a config.

Tile size and number of splits (segments) are derived from the following parameters:
TILE_SIZE_MIN/MAX, and MIN_SEGMENTS/MAX_SEGMENTS/SEGMENTS_PER_CU, plus MFMA_DIM
and the SPLIT_MIN_* floors; see compute_segment_params.
"""

import copy
import functools
import itertools
import os

import torch
import triton
from torch.overrides import TorchFunctionMode

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.config_utils import (
    USE_LRU_CACHE,
    load_config_json,
    resolve_config_dir,
)
from aiter.ops.triton.utils.types import e4m3_dtype

_CONFIG_NAME = "UNIFIED-ATTENTION"
_OPS = ("attn_2d", "attn_3d", "reduce", "kv_split")

# Groups of RDNA GPUs have similar hardware. Use the same config file until further investigation shows they diverge.
_ARCH_ALIAS = {
    "gfx1101": "gfx1100",
    "gfx1102": "gfx1100",
}

_SEP = "."

_AXIS_KIND = {
    "D": "num",  # head_size
    "Q": "num",  # max_seqlen_q; Q_LEQ_1 is decode
    "BS": "num",  # block_size, the KV page size
    "SW": "opt_num",  # keys a row can see, 0 when off; bare SW = any window
    "SKLT": "bool",  # max_seqlen_k < 2048
    "SHUF": "bool",  # shuffled_kv_cache
    "DT": "enum",  # (q dtype tag, kv dtype tag)
    # 2d programs per CU, rounded up to a power of 2, so P bounds must be powers of 2
    "P": "num",
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
    # every matching LEQ key outranks every GEQ key, so one direction per axis
    for a in axes:
        if _AXIS_KIND[a] == "opt_num":
            dirs = {p.split("_")[1] for p in parts[a] if "_" in p}
            assert not {"LEQ", "GEQ"} <= dirs, (
                f"{_CONFIG_NAME}: {a} keys mix LEQ and GEQ "
                f"({sorted(parts[a])}); use one direction"
            )
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
    if kind == "opt_num" and not value:
        return ["any"]

    eq = sorted(c for c in parts if c.startswith(f"{axis}_EQ_"))
    leq = sorted((c for c in parts if c.startswith(f"{axis}_LEQ_")), key=_bound)
    geq = sorted(
        (c for c in parts if c.startswith(f"{axis}_GEQ_")), key=_bound, reverse=True
    )
    return (
        [c for c in eq if value == _bound(c)]
        + [c for c in leq if value <= _bound(c)]
        + [c for c in geq if value >= _bound(c)]
        + ([axis] if kind == "opt_num" else [])
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
    mfma_dim = config.pop("MFMA_DIM", None)
    min_tiles = config.pop("SPLIT_MIN_TILES", 0)
    min_share = config.pop("SPLIT_MIN_SHARE", 0)
    fit_budget = config.pop("SPLIT_FIT_BUDGET", False)
    seqs_per_tile = config.pop("SPLIT_SEQS_PER_TILE", 0)

    # tokens one segment must cover, so the split never outruns the context
    tile = triton.next_power_of_2(params.block_size)
    tile = max(tile_lo, tile if tile_hi is None else min(tile_hi, tile))
    kv_len = params.max_seqlen_k
    # a sliding-window split carves only the window: rows see at most
    # window + q - 1 keys
    if params.sliding_window > 0:
        kv_len = min(kv_len, params.sliding_window + params.max_seqlen_q - 1)
    limit = triton.cdiv(kv_len, tile)
    too_many_seqs = seqs_per_tile and params.num_seqs > seqs_per_tile * limit
    if cap is not None:
        limit = min(cap, limit)

    budget = params.num_sms * per_cu
    prgms = max(1, params.num_2d_prgms)
    # this is specific to gfx950 gluon as the num waves depends on mfma dim there
    if mfma_dim:
        num_waves = max(
            1, triton.next_power_of_2(params.num_queries_per_kv) // mfma_dim
        )
        budget //= num_waves
    share = triton.cdiv(budget, prgms)
    if fit_budget:
        share = 1 << ((budget // prgms).bit_length() - 1) if budget >= prgms else 0
    if limit <= min_tiles or share < min_share or too_many_seqs:
        segments = 1
    else:
        claim = max(min(lo, limit), min(limit, max(1, share)))
        segments = triton.next_power_of_2(claim)
    if small_split_max is None:
        config["NUM_SEGMENTS"] = segments
    elif segments <= min(small_split_max, limit):
        # the split landed on its floor: too few segments to be worth 2 warps
        config["num_warps"] = 1
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
    programs_per_cu,
) -> dict:
    return {
        "D": head_size,
        "Q": max_seqlen_q,
        "BS": block_size,
        "SW": max(1, min(sliding_window, max_seqlen_k)) if sliding_window > 0 else 0,
        "SKLT": max_seqlen_k < 2048,
        "SHUF": shuffled_kv_cache,
        "DT": (get_dtype_str(q_dtype), get_dtype_str(kv_dtype)),
        "P": programs_per_cu,
    }


def _programs_per_cu(params) -> int:
    return triton.next_power_of_2(triton.cdiv(params.num_2d_prgms, params.num_sms))


def _load(op: str, backend, arch: str | None) -> tuple:
    """Return ``(table, axes, cfg_dir)`` for one op."""
    if arch is None:
        arch = arch_info.get_arch()
    cfg_dir = resolve_config_dir(
        "attention", _CONFIG_NAME, backend=backend, arch=_ARCH_ALIAS.get(arch, arch)
    )
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
    return config[op], axes, cfg_dir


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
    programs_per_cu: int,
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
        programs_per_cu,
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
        # no asymmetric-head configs yet, pick by the wider of the two heads
        max(params.head_size, params.head_size_v),
        params.max_seqlen_q,
        params.max_seqlen_k,
        params.sliding_window,
        params.shuffled_kv_cache,
        params.q_dtype,
        params.kv_cache_dtype,
        params.block_size,
        _programs_per_cu(params),
        backend,
        arch,
    )
    # NUM_SEGMENTS is derived out here rather than in the cached lookup: it
    # reads the launch's program count, which is not part of the cache key.
    return compute_segment_params(copy.deepcopy(config), params)


def explain(op: str, params, backend: str = "triton", arch: str | None = None) -> str:
    """Report which entry a lookup lands on, and the config it yields."""
    table, axes, cfg_dir = _load(op, backend, arch)
    values = _axis_values(
        max(params.head_size, params.head_size_v),
        params.max_seqlen_q,
        params.max_seqlen_k,
        params.sliding_window,
        params.shuffled_kv_cache,
        params.block_size,
        params.q_dtype,
        params.kv_cache_dtype,
        _programs_per_cu(params),
    )
    key, config = _lookup(table, axes, values)
    derived = compute_segment_params(
        compute_tile_params(dict(config), params.block_size), params
    )

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


# -------------------------------------------------------------------------------------------------------------------
# AITER_UA_FAST_LAUNCH=1: record-and-replay host fast path for eager `unified_attention`.
#
# The eager call (a prefill's attention layer, say) spends ~45-110 us on the host: routing, config lookups and copies,
# building _UAParams and ~75 kernel kwargs, then Triton's JITFunction.run per kernel (binder, cache key, launch). A
# step calls it once per full-attention layer with the same shapes, so the work is the same every layer.
#
# The first call with a given key runs the normal path while recording what it does: each `torch.empty`, and each
# Triton launch (the compiled kernel, its grid, its full argument list, and for every tensor argument whether it is
# input tensor i or allocation j). Later calls with the same key allocate the same buffers and call each compiled
# kernel's launcher directly, with this call's tensors substituted in.
#
# The key holds everything the recorded path can depend on: every scalar argument by value; for every tensor argument
# its dtype, shape, strides, device, 16-byte pointer alignment and (with AMD buffer ops) whether its storage is within
# 2 GB -- the per-argument facts Triton specializes a kernel on -- and which tensor arguments are the same object.
# A call that does anything else on the host (a torch op other than metadata reads and torch.empty, a tensor argument
# that is neither an input nor one of its allocations, a launch hook) is not recorded and keeps the normal path.
# Never used while a CUDA graph is being captured.
# -------------------------------------------------------------------------------------------------------------------

_MAX_INT_32 = 2**31 - 1
_MAX_PLANS = int(os.environ.get("AITER_UA_FAST_LAUNCH_MAX_PLANS", "4096"))
_NOT_REPLAYABLE = object()

_LOG = os.environ.get("AITER_UA_FAST_LAUNCH_LOG", "0") == "1"
_plans: dict = {}
fast_launch_stats = {"hit": 0, "miss": 0, "not_replayable": 0, "bypass": 0}

_rt = None


def _runtime():
    global _rt
    if _rt is None:
        import triton.runtime.jit as tj
        from triton import knobs
        from triton.runtime.driver import driver

        _rt = (tj, knobs, driver.active)
    return _rt


class _Unreplayable(Exception):
    pass


# Tensor methods the recorded path may call: metadata only, no device work, no data reads.
_METADATA = {
    torch.Tensor.dim,
    torch.Tensor.stride,
    torch.Tensor.size,
    torch.Tensor.data_ptr,
    torch.Tensor.untyped_storage,
    torch.Tensor.element_size,
    torch.Tensor.numel,
    torch.Tensor.is_contiguous,
    torch.Tensor.storage_offset,
    torch.Tensor.__len__,
    torch.Tensor.get_device,
}


def _flatten(obj, tensors, buffer_ops):
    """Key description of one argument; appends its tensors (in order) to `tensors`."""
    if isinstance(obj, torch.Tensor):
        tensors.append(obj)
        ptr = obj.data_ptr()
        within = obj.untyped_storage().size() <= _MAX_INT_32 if buffer_ops else False
        return (obj.dtype, obj.shape, obj.stride(), obj.device, ptr % 16 == 0, within)
    if isinstance(obj, (tuple, list)):
        return (type(obj), tuple(_flatten(x, tensors, buffer_ops) for x in obj))
    return obj  # scalars, None, strings, dtypes: hashable values


def _make_key(args, kwargs, buffer_ops):
    tensors = []
    desc = [_flatten(a, tensors, buffer_ops) for a in args]
    kdesc = [(n, _flatten(v, tensors, buffer_ops)) for n, v in kwargs.items()]
    # which tensor arguments are the same object: the recording maps a launch argument to the first one
    first = {}
    alias = tuple(first.setdefault(id(t), i) for i, t in enumerate(tensors))
    return (tuple(desc), tuple(kdesc), alias), tensors


class _Recorder(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.allocs = []  # (args, kwargs) of each torch.empty, in order
        self.alloc_ids = {}  # id(result) -> allocation index
        self.alloc_keep = []  # keep results alive so ids stay unique while recording
        self.bad = None

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.empty:
            out = func(*args, **kwargs)
            self.alloc_ids[id(out)] = len(self.allocs)
            self.allocs.append((args, dict(kwargs)))
            self.alloc_keep.append(out)
            return out
        if func in _METADATA or getattr(func, "__name__", "") == "__get__":
            return func(*args, **kwargs)
        if self.bad is None:
            self.bad = f"torch op {getattr(func, '__qualname__', func)}"
        return func(*args, **kwargs)


def _record(impl, args, kwargs, tensors):
    """Run the normal path, recording it. Returns (result, plan or _NOT_REPLAYABLE)."""
    tj, knobs, drv = _runtime()
    rec = _Recorder()
    launches = []
    orig_run = tj.JITFunction.run

    def run(self, *a, grid, warmup, **kw):
        kernel = orig_run(self, *a, grid=grid, warmup=warmup, **kw)
        if warmup or kernel is None:
            rec.bad = rec.bad or "warmup or failed launch"
            return kernel
        if self.pre_run_hooks or knobs.runtime.add_stages_inspection_hook is not None:
            rec.bad = rec.bad or "triton hooks"
            return kernel
        # the same options JITFunction.run hands its binder
        kw = dict(kw)
        kw["debug"] = kw.get("debug", self.debug) or knobs.runtime.debug
        kw["instrumentation_mode"] = knobs.compilation.instrumentation_mode
        kw["fpsan_homomorphic_casts"] = knobs.compilation.fpsan_homomorphic_casts
        device = drv.get_current_device()
        binder = self.device_caches[device][4]
        bound_args, _, _ = binder(*a, **kw)
        g = grid(bound_args) if callable(grid) else grid
        g = (g[0], g[1] if len(g) > 1 else 1, g[2] if len(g) > 2 else 1)
        launches.append((self, kernel, g, list(bound_args.values())))
        return kernel

    tj.JITFunction.run = run
    try:
        with rec:
            result = impl(*args, **kwargs)
    finally:
        tj.JITFunction.run = orig_run
    if rec.bad is not None:
        bad = rec.bad
        rec.alloc_keep.clear()
        return result, (_NOT_REPLAYABLE, bad)

    in_ids = {}
    for i, t in enumerate(tensors):
        in_ids.setdefault(id(t), i)
    alloc_ids = rec.alloc_ids
    allocs = tuple(rec.allocs)
    # Drop the recorder's references to this call's buffers now: nothing below may keep them alive.
    rec.alloc_keep.clear()
    try:
        plan_launches = []
        for fn, kernel, g, vals in launches:
            subs = []
            template = list(vals)
            for pos, v in enumerate(vals):
                if isinstance(v, torch.Tensor):
                    subs.append((pos,) + _source(v, in_ids, alloc_ids))
                    template[pos] = None
            plan_launches.append((fn, kernel, g, template, tuple(subs)))
        plan = (allocs, tuple(plan_launches), _ret_spec(result, in_ids, alloc_ids))
    except _Unreplayable as e:
        return result, (_NOT_REPLAYABLE, str(e))
    return result, plan


# Module-level, not closures: a recursive closure is a reference cycle, and a cycle holding a call's buffers frees
# them only when the cyclic GC happens to run -- which in a server is long after the GPU memory has run out.
def _source(x, in_ids, alloc_ids):
    if id(x) in in_ids:
        return (0, in_ids[id(x)])
    if id(x) in alloc_ids:
        return (1, alloc_ids[id(x)])
    raise _Unreplayable(
        "a tensor that is neither an input nor an allocation of the call"
    )


def _ret_spec(r, in_ids, alloc_ids):
    if r is None:
        return ("none",)
    if isinstance(r, torch.Tensor):
        return ("t",) + _source(r, in_ids, alloc_ids)
    if isinstance(r, tuple):
        return ("tuple", tuple(_ret_spec(x, in_ids, alloc_ids) for x in r))
    raise _Unreplayable(f"return value of type {type(r).__name__}")


def _build(spec, srcs):
    kind = spec[0]
    if kind == "none":
        return None
    if kind == "t":
        return srcs[spec[1]][spec[2]]
    return tuple(_build(s, srcs) for s in spec[1])


def _replay(plan, tensors):
    _, knobs, drv = _runtime()
    allocs_spec, launches, ret = plan
    allocs = [torch.empty(*a, **kw) for a, kw in allocs_spec]
    srcs = (tensors, allocs)
    rt = knobs.runtime
    enter, exit_ = rt.launch_enter_hook, rt.launch_exit_hook
    device = drv.get_current_device()
    stream = drv.get_current_stream(device)
    not_present = object()
    for fn, kernel, g, template, subs in launches:
        for (name, _), (val, globals_dict) in fn.used_global_vals.items():
            if globals_dict.get(name, not_present) != val:
                raise RuntimeError(
                    f"Global variable {name} has changed since we compiled this kernel"
                )
        args = template.copy()
        for pos, which, idx in subs:
            args[pos] = srcs[which][idx]
        kernel.run(
            g[0],
            g[1],
            g[2],
            stream,
            kernel.function,
            kernel.packed_metadata,
            kernel.launch_metadata(g, stream, *args),
            enter,
            exit_,
            *args,
        )

    return _build(ret, srcs)


def wrap_fast_launch(impl):
    """`unified_attention` with the fast path in front of `impl`."""

    @functools.wraps(impl)
    def unified_attention(*args, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            fast_launch_stats["bypass"] += 1
            return impl(*args, **kwargs)
        _, knobs, _ = _runtime()
        key, tensors = _make_key(args, kwargs, knobs.amd.use_buffer_ops)
        try:
            plan = _plans.get(key)
        except TypeError:  # an unhashable argument: no key
            fast_launch_stats["bypass"] += 1
            return impl(*args, **kwargs)
        if plan is None:
            fast_launch_stats["miss"] += 1
            result, plan = _record(impl, args, kwargs, tensors)
            if len(_plans) >= _MAX_PLANS:
                _plans.clear()
            _plans[key] = plan
            if _LOG and plan[0] is _NOT_REPLAYABLE:
                print(f"[aiter UA fast launch] not replayable: {plan[1]}", flush=True)
            return result
        if plan[0] is _NOT_REPLAYABLE:
            fast_launch_stats["not_replayable"] += 1
            return impl(*args, **kwargs)
        fast_launch_stats["hit"] += 1
        if _LOG and fast_launch_stats["hit"] % 2000 == 0:
            print(
                f"[aiter UA fast launch] pid {os.getpid()} {fast_launch_stats} plans {len(_plans)} "
                f"mem alloc {torch.cuda.memory_allocated() / 2**30:.2f} GiB max {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
                f"reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB",
                flush=True,
            )
        return _replay(plan, tensors)

    unified_attention._fast_launch_impl = impl
    return unified_attention
