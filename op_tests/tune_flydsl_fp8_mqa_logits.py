# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Search-driven tuning of the gfx950 FlyDSL fp8_mqa_logits kernel.

Plugs the kernel builders into ``aiter.utility.tuning_search``: every builder
knob, the ``waves_per_eu`` compile hint and the runtime ``grid.y`` KV split
form one gated search space, and a ``CallableBackend`` compiles, grades and
times a config on one problem (shape, heads, head dim, clean_logits, window).

Measurement matches ``test_flydsl_fp8_mqa_logits.py --scenario bench``: HIP
graph replays, each preceded by an LLC flush outside the timed bracket, and
the same NaN-poisoned, -inf-mask-exact ``calc_diff`` grading -- on the tuned
problem plus three small edge problems (window tails, empty and past-the-end
windows) before a config's first timing.

Single problem, in process::

    python op_tests/tune_flydsl_fp8_mqa_logits.py -p 4096,4096,64,64,0,causal

Several problems, one worker process per GPU::

    python op_tests/tune_flydsl_fp8_mqa_logits.py --gpus 0,1,2,3 \\
        -p 4096,4096,64,64,0,causal 1024,1024,32,128,0,misaligned

Through the engine's CLI (one problem)::

    python -m aiter.utility.tuning_search tune \\
        --factory op_tests.tune_flydsl_fp8_mqa_logits:make \\
        --arg problem=4096,4096,64,64,0,causal --budget 128
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import queue
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import namedtuple
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from aiter.utility.tuning_search import (
    Budget,
    CompileOutcome,
    Fidelity,
    Param,
    SearchSpace,
    TuningDB,
    WarmStart,
    record_result,
    tune,
)

Problem = namedtuple("Problem", "s_q s_k num_heads head_dim clean_logits window")

# gfx950 LDS per CU.
_LDS_BYTES = 160 * 1024
# Small problems every config is graded on besides the tuned one: a short
# odd tail, rows with empty windows (both spellings), cu_starts past the end,
# windows straddling tile edges, and a batch_causal window whose last row
# reaches the final column of an odd seq_len_kv. No other window here covers
# that column (the test's causal mode stops one short), and with
# clean_logits=False nothing else would grade it.
_EDGE_PROBLEMS = (
    (37, 1001, "batch_causal"),
    (17, 76, "past_end"),
    (61, 113, "empty"),
    (257, 1000, "misaligned"),
)
_DEFAULT_OUT = "op_tests/dump_data/tuning/fp8_mqa_logits"
_SPILL_KEYS = ("spill_vgpr", "spill_sgpr", "private_segment_size")


def parse_problem(text: str) -> Problem:
    """``s_q,s_k,num_heads,head_dim,clean_logits,window``; window ``bc<B>`` is
    the block-diagonal ``batch_causal`` window of ``B`` packed sequences."""
    s_q, s_k, h, d, clean, window = text.split(",")
    return Problem(int(s_q), int(s_k), int(h), int(d), bool(int(clean)), window)


def problem_tag(p: Problem) -> str:
    return (
        f"{p.s_q}x{p.s_k}_H{p.num_heads}_D{p.head_dim}"
        f"_{'cl' if p.clean_logits else 'nocl'}_{p.window}"
    )


# --------------------------------------------------------------------- space
_MFMA_SHAPES = ("32x32x64", "16x16x128")
_HEAD_ACC = ("add", "fma", "fma2")
_MFMA_DIMS = {"32x32x64": (32, 32, 64), "16x16x128": (16, 16, 128)}


def is_valid(cfg: dict, num_heads: int, head_dim: int) -> bool:
    """Mirror of the builder asserts plus the LDS budget, so the search never
    spends a compile on a config the builder would refuse."""
    m, n, k = _MFMA_DIMS[cfg["mfma"]]
    bkv, wpb = cfg["block_kv"], cfg["waves_per_block"]
    if num_heads % m or bkv % n or head_dim % k:
        return False
    n_tiles = bkv // n
    if not cfg["lds"]:
        return n_tiles % wpb == 0
    nb, pd = cfg["num_buffers"], cfg["prefetch_depth"]
    if nb < pd:
        return False
    kv_bytes = bkv * head_dim
    if kv_bytes % (64 * wpb * 16):
        return False
    scale_dmas = -(-bkv // 64) if cfg["lds_scales"] else 0
    if (pd - 1) * (kv_bytes // (64 * wpb * 16) + scale_dmas) > 63:
        return False
    if cfg["rs_head"] and n_tiles % (64 // n):
        return False
    return nb * (kv_bytes + scale_dmas * 64 * 4) <= _LDS_BYTES


def make_space(num_heads: int, head_dim: int, auto_split: bool = False) -> SearchSpace:
    """Builder knobs (LDS-pipe sub-knobs gated on ``lds``), the
    ``waves_per_eu`` hint, and the KV split: ``auto`` (the launcher's policy)
    or a fixed ``num_splits``. ``auto_split`` pins the split to the policy, for
    tuning tags that are routed without a ``_ns``."""
    params = (
        Param.categorical("mfma", _MFMA_SHAPES),
        Param.ordinal("block_kv", (64, 128, 256)),
        # Rows per wave on the LDS path (a block owns rows * waves), rows per
        # block on the direct-load path.
        Param.ordinal("rows_per_block", (1, 2, 4)),
        Param.ordinal("waves_per_block", (1, 2, 4)),
        Param.boolean("lds"),
        Param.ordinal("num_buffers", (2, 3, 4), gate="lds"),
        Param.ordinal("prefetch_depth", (1, 2, 3), gate="lds"),
        Param.boolean("sw_pipe", gate="lds"),
        Param.boolean("rs_head", gate="lds"),
        Param.boolean("lds_scales", gate="lds"),
        Param.boolean("tile_skip", gate="lds"),
        Param.boolean("dma_guard", gate="lds"),
        Param.categorical("head_acc", _HEAD_ACC),
        Param.ordinal("waves_per_eu", (1, 2, 3, 4)),
        Param.categorical("split", ("auto",) if auto_split else ("auto", "fixed")),
        Param.ordinal("num_splits", (1, 2, 4, 8, 16, 32, 64), gate="split"),
    )
    return SearchSpace(
        params,
        is_valid=lambda c: is_valid(c, num_heads, head_dim),
        name=f"fp8_mqa_logits_gfx950_H{num_heads}_D{head_dim}",
    )


def config_from_tag(tag: str) -> dict:
    """The knob dict of a gfx950 variant tag (the launcher's tag grammar, see
    ``fp8_mqa_logits_kernels``); absent segments take the launcher defaults."""
    from aiter.ops.flydsl.fp8_mqa_logits_kernels import _TAG_RE

    m = _TAG_RE.match(tag)
    if m is None or not m["shape"] or not m["bkv"]:
        raise ValueError(f"not a gfx950 variant tag: {tag!r}")
    cfg = {
        "mfma": m["shape"],
        "block_kv": int(m["bkv"]),
        "rows_per_block": int(m["rpb"]),
        "waves_per_block": int(m["wpb"]),
        "lds": m["lds"] is not None,
        "head_acc": m["ha"] or "add",
        "waves_per_eu": int(m["wpe"] or 2),
        "split": "fixed" if m["ns"] else "auto",
    }
    if m["ns"]:
        cfg["num_splits"] = int(m["ns"])
    if cfg["lds"]:
        cfg.update(
            num_buffers=int(m["lds"]),
            prefetch_depth=int(m["pd"] or 2),
            sw_pipe=bool(m["swp"]),
            rs_head=bool(m["rs"]),
            lds_scales=bool(m["ls"]),
            tile_skip=bool(m["ts"]),
            dma_guard=bool(m["dg"]),
        )
    return cfg


def config_label(cfg: dict) -> str:
    """The launcher's variant tag for a knob dict (inverse of
    :func:`config_from_tag`; default segments are left out)."""
    s = f"mfma{cfg['mfma']}_bkv{cfg['block_kv']}_r{cfg['rows_per_block']}"
    s += f"_w{cfg['waves_per_block']}"
    if cfg["lds"]:
        s += f"_lds{cfg['num_buffers']}"
        if cfg["prefetch_depth"] != 2:
            s += f"_pd{cfg['prefetch_depth']}"
        s += "_swp" * cfg["sw_pipe"] + "_rs" * cfg["rs_head"]
        s += "_ls" * cfg["lds_scales"]
        s += "_ts" * cfg.get("tile_skip", False)
        s += "_dg" * cfg.get("dma_guard", False)
    if cfg.get("head_acc", "add") != "add":
        s += f"_h{cfg['head_acc']}"
    if cfg["waves_per_eu"] != 2:
        s += f"_wpe{cfg['waves_per_eu']}"
    if cfg["split"] == "fixed":
        s += f"_ns{cfg['num_splits']}"
    return s


# ------------------------------------------------------------------- backend
def _on_cuda(fn):
    """Run with CUDA as the default device. The evaluator calls the backend
    from a worker thread, and ``torch.set_default_device`` is thread-local."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with torch.device("cuda"):
            return fn(*args, **kwargs)

    return wrapped


def _kernel_resources(launcher, args) -> dict:
    """Compile only (meta tensors) and read the AMDGPU kernel metadata."""
    import flydsl.compiler as flyc

    meta = [a.to("meta") if isinstance(a, torch.Tensor) else a for a in args]
    ir = flyc.compile(launcher, *meta)._keepalive._ir_text
    m = re.search(r"kernel_metadata<\"[^\"]+\",[^{]*\{([^}]*)\}", ir)
    if m is None:
        return {}
    attrs = dict(re.findall(r"(\w+) = (\d+) : i64", m.group(1)))
    get = lambda k: float(attrs.get(k, 0))
    return {
        "vgpr": get("vgpr_count"),
        "agpr": get("agpr_count"),
        "sgpr": get("sgpr_count"),
        "lds_bytes": get("group_segment_fixed_size"),
        "spill_vgpr": get("vgpr_spill_count"),
        "spill_sgpr": get("sgpr_spill_count"),
        "private_segment_size": get("private_segment_fixed_size"),
    }


class Fp8MqaLogitsBackend:
    """Compile / check / measure for one problem on the current device.

    ``replays`` graph replays (each after an LLC flush) are averaged into one
    sample, and a fidelity's ``reps`` is the number of samples, so the search
    sees the same statistic as the bench.
    """

    def __init__(self, problem: Problem, *, replays: int = 25):
        from op_tests import test_flydsl_fp8_mqa_logits as t

        self.t = t
        self.p = problem
        self.replays = replays
        self.inputs = {}
        self.graphs = {}
        self.cache = t._l2_flush_cache()
        self.stream = torch.cuda.Stream()

    # -- inputs and launches -------------------------------------------------
    def _inputs(self, s_q, s_k, window):
        key = (s_q, s_k, window)
        if key not in self.inputs:
            batch = 1
            if window.startswith("bc"):
                window, batch = "batch_causal", int(window[2:])
            self.inputs[key] = self.t._make_inputs(
                s_q, s_k, self.p.num_heads, self.p.head_dim, "fn", "fn", window, batch
            )
        return self.inputs[key]

    def _launch_args(self, inp, out):
        s_q, s_k = inp.q_fp8.shape[0], inp.kv_fp8.shape[0]
        tensors = (inp.q_fp8, inp.kv_fp8, inp.scales, inp.weights, inp.ks, inp.ke, out)
        return tensors, (s_q, s_k, int(out.stride(0)))

    def _num_splits(self, cfg, launcher, tensors, scalars) -> int:
        from aiter.ops.flydsl import fp8_mqa_logits_kernels as k

        if cfg["split"] == "fixed":
            return int(cfg["num_splits"])
        # The launcher's own policy, on the tag of this config.
        return k._policy_num_splits(
            launcher,
            (*tensors, *scalars, 1, self.stream),
            config_label(cfg),
            scalars[0],
            scalars[1],
            self.p.num_heads,
            torch.cuda.current_device(),
        )

    def _runner(self, cfg, launcher, inp, out):
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

        tensors, scalars = self._launch_args(inp, out)
        ns = self._num_splits(cfg, launcher, tensors, scalars)
        return lambda: _run_compiled(launcher, *tensors, *scalars, ns, self.stream)

    @staticmethod
    def _out(inp):
        s_q, s_k = inp.q_fp8.shape[0], inp.kv_fp8.shape[0]
        # The launcher's 256-column aligned row stride.
        out = torch.empty((s_q, -(-s_k // 256) * 256), dtype=torch.float32)
        return out[:, :s_k]

    # -- Backend protocol ----------------------------------------------------
    def environment(self) -> dict:
        return {"problem": problem_tag(self.p), "replays": self.replays}

    @_on_cuda
    def compile(self, cfg, key):
        from aiter.ops.flydsl import fp8_mqa_logits_kernels as k
        from aiter.ops.flydsl.kernels.mqa_logits import fp8_mqa_logits as b

        atom = {"32x32x64": b._MFMA32_K64, "16x16x128": b._MFMA16_K128}[cfg["mfma"]]
        common = {
            "num_heads": self.p.num_heads,
            "head_size": self.p.head_dim,
            "block_kv": cfg["block_kv"],
            "rows_per_block": cfg["rows_per_block"],
            "waves_per_block": cfg["waves_per_block"],
            "mfma": atom,
            "clean_logits": self.p.clean_logits,
            "head_acc": cfg.get("head_acc", "add"),
        }
        try:
            if cfg["lds"]:
                launcher = b._build_kernel_mfma_lds_pipe(
                    **common,
                    num_buffers=cfg["num_buffers"],
                    prefetch_depth=cfg["prefetch_depth"],
                    sw_pipe=cfg["sw_pipe"],
                    rs_head=cfg["rs_head"],
                    lds_scales=cfg["lds_scales"],
                    tile_skip=cfg.get("tile_skip", False),
                    dma_guard=cfg.get("dma_guard", False),
                )
            else:
                launcher = b._build_kernel_mfma_r_w(**common)
        except AssertionError as e:
            return CompileOutcome(False, detail=f"builder: {e}")
        launcher.compile_hints = {
            **k._DEFAULT_COMPILE_HINTS,
            "waves_per_eu": cfg["waves_per_eu"],
        }
        inp = self._inputs(self.p.s_q, self.p.s_k, self.p.window)
        tensors, scalars = self._launch_args(inp, self._out(inp))
        resources = _kernel_resources(launcher, (*tensors, *scalars, 1, self.stream))
        return CompileOutcome(True, artifact=launcher, resources=resources)

    def _grade(self, cfg, launcher, s_q, s_k, window) -> str | None:
        inp = self._inputs(s_q, s_k, window)
        out = self._out(inp).fill_(float("nan"))
        # The poison fill runs on the current stream; without this wait it can
        # land after the kernel and fail a correct config.
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self._runner(cfg, launcher, inp, out)()
        self.stream.synchronize()
        try:
            self.t._grade_all(
                {"flydsl": out},
                inp,
                s_q,
                s_k,
                self.p.num_heads,
                self.p.clean_logits,
                f"{s_q}x{s_k} {window}",
            )
        except AssertionError as e:
            return str(e)
        return None

    @_on_cuda
    def check(self, cfg, key, launcher):
        problems = ((self.p.s_q, self.p.s_k, self.p.window), *_EDGE_PROBLEMS)
        for s_q, s_k, window in problems:
            why = self._grade(cfg, launcher, s_q, s_k, window)
            if why is not None:
                return False, why
        return True, ""

    def _graph(self, key, fn):
        if key in self.graphs:
            return self.graphs[key]
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            fn()
        # Graphs pin their buffers; keep only the recent ones.
        if len(self.graphs) > 64:
            self.graphs.pop(next(iter(self.graphs)))
        self.graphs[key] = graph
        return graph

    def _sample_us(self, graph) -> float:
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(self.replays)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(self.replays)]
        for i in range(self.replays):
            self.cache.zero_()
            starts[i].record()
            graph.replay()
            ends[i].record()
        torch.cuda.synchronize()
        return statistics.mean(s.elapsed_time(e) for s, e in zip(starts, ends)) * 1e3

    def _config_graph(self, cfg, key, launcher):
        def make():
            inp = self._inputs(self.p.s_q, self.p.s_k, self.p.window)
            return self._runner(cfg, launcher, inp, self._out(inp))

        if key not in self.graphs:
            return self._graph(key, make())
        return self.graphs[key]

    @_on_cuda
    def measure(self, cfg, key, launcher, fidelity):
        graph = self._config_graph(cfg, key, launcher)
        for _ in range(fidelity.warmup):
            self._sample_us(graph)
        return [self._sample_us(graph) for _ in range(fidelity.reps)]

    @_on_cuda
    def measure_batch(self, items, fidelity):
        """Samples interleaved across ``items`` in a shuffled order per rep."""
        graphs = [self._config_graph(c, k, a) for c, k, a in items]
        for g in graphs:
            for _ in range(fidelity.warmup):
                self._sample_us(g)
        out = [[] for _ in items]
        order = list(range(len(items)))
        rng = random.Random(len(items))
        for _ in range(fidelity.reps):
            rng.shuffle(order)
            for i in order:
                out[i].append(self._sample_us(graphs[i]))
        return out

    @_on_cuda
    def reference_us(self, reps: int = 15) -> float:
        """The Gluon / Triton kernel on this problem, same timer."""
        inp = self._inputs(self.p.s_q, self.p.s_k, self.p.window)
        fn = lambda: self.t.triton_logits(
            inp.q_fp8,
            inp.kv_fp8,
            inp.scales,
            inp.weights,
            inp.ks,
            inp.ke,
            self.p.clean_logits,
        )
        graph = self._graph("__reference__", fn)
        return statistics.median(self._sample_us(graph) for _ in range(reps))


# ------------------------------------------------------------------ factory
def make(args) -> dict:
    """``python -m aiter.utility.tuning_search tune --factory`` entry point:
    ``--arg problem=s_q,s_k,H,D,clean,window`` (``replays=N`` optional)."""
    fa = getattr(args, "factory_args", {}) or {}
    p = parse_problem(fa["problem"])
    from aiter.ops.flydsl.fp8_mqa_logits_kernels import _auto_variant

    space = make_space(p.num_heads, p.head_dim)
    auto = config_from_tag(_auto_variant(p.s_q, p.s_k, p.num_heads, p.head_dim))
    return {
        "space": space,
        "backend": Fp8MqaLogitsBackend(p, replays=int(fa.get("replays", 25))),
        "baseline": space.encode(auto),
        "record": lambda res: {
            "family": "fp8_mqa_logits",
            "candidate": "flydsl",
            "arch": "gfx950",
            "problem": problem_tag(p),
            "tuning_id": f"{problem_tag(p)}@{res.best_key}",
            "config_key": res.best_key,
            "knobs": res.best_config,
            "unit": "us",
        },
    }


# ------------------------------------------------------------------- driver
def tune_problem(p: Problem, opts) -> dict:
    from aiter.ops.flydsl.fp8_mqa_logits_kernels import _auto_variant

    out_dir = Path(opts.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = problem_tag(p)
    space = make_space(p.num_heads, p.head_dim, opts.auto_split)
    backend = Fp8MqaLogitsBackend(p, replays=opts.replays)
    auto_tag = _auto_variant(p.s_q, p.s_k, p.num_heads, p.head_dim)
    auto = config_from_tag(auto_tag)
    baseline = space.encode(auto)
    warm = [WarmStart(config_from_tag(t), source="registry") for t in opts.warm]
    warm += _warm_from(opts.warm_from, p)
    if opts.auto_split:
        warm = _auto_split_warm(warm)
    cache_dir = Path(opts.cache_dir or out_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = tune(
        space,
        backend,
        strategy=opts.strategy,
        budget=Budget(evaluations=opts.budget),
        seed=opts.seed,
        ladder=(Fidelity("full", reps=opts.reps, warmup=1),),
        warm=[*warm],
        batch=opts.batch,
        baseline=baseline,
        trace=out_dir / f"{tag}.trace.jsonl",
        cache_path=cache_dir / f"{tag}.cache.json",
        compile_timeout=600,
        measure_timeout=120,
        confirm=4,
        confirm_rounds=3,
        # A hill climb starts from the auto pick, or from the first warm start.
        strategy_options=(
            {"start": baseline} if opts.strategy == "hill" and not warm else None
        ),
    )
    conf = res.confirmed or {}
    # A fresh interleaved A/B of the auto pick against the winner, outside the
    # search's own samples, plus the reference kernel on the same timer.
    final = {}
    if res.best_key:
        items = []
        for name, cfg in (("auto", auto), ("best", res.best_config)):
            comp = backend.compile(cfg, name)
            items.append((cfg, name, comp.artifact))
        got = backend.measure_batch(items, Fidelity("final", reps=2 * opts.reps))
        final = {k: statistics.median(v) for (_c, k, _a), v in zip(items, got)}
    base_us = final.get("auto")
    best_us = final.get("best")
    ref_us = backend.reference_us()
    summary = {
        "problem": p._asdict(),
        "tag": tag,
        "strategy": res.strategy,
        "evaluations": res.evaluations,
        "seconds": time.time() - t0,
        "auto_variant": auto_tag,
        "auto_us": base_us,
        "best": config_label(res.best_config) if res.best_key else None,
        "best_config": res.best_config,
        "best_us": best_us,
        "search_best_us": res.best_value if math.isfinite(res.best_value) else None,
        "reference_us": ref_us,
        "confirmed_speedup_vs_auto": conf.get("speedup"),
        "p_value_vs_auto": conf.get("p_value"),
        "speedup_vs_auto": base_us / best_us if best_us else None,
        "speedup_vs_reference": ref_us / best_us if best_us else None,
        "auto_vs_reference": ref_us / base_us if base_us else None,
        "failures": sum(1 for ob in res.observations if not ob.ok),
    }
    (out_dir / f"{tag}.json").write_text(json.dumps(summary, indent=2, default=str))
    if res.best_key:
        record_result(
            res,
            family="fp8_mqa_logits",
            candidate="flydsl",
            arch="gfx950",
            problem=tag,
            tuning_id=f"{tag}@{res.best_key}",
            config_key=res.best_key,
            knobs=res.best_config,
            db=TuningDB(out_dir / f"{tag}.db.json"),
            unit="us",
            seed=opts.seed,
        )
    return summary


def _load_summaries(dirs):
    rows = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.json")):
            if not f.name.endswith((".cache.json", ".db.json", ".eval.json")):
                rows.append(json.loads(f.read_text()))
    return rows


def _expand_new_knobs(cfg: dict) -> list[dict]:
    """A winner from before a knob existed, under each of the knob's values
    (the default first): the lookback after a base change."""
    out = [cfg]
    if "head_acc" not in cfg:
        out = [{**c, "head_acc": h} for c in out for h in _HEAD_ACC]
    for knob in ("tile_skip", "dma_guard"):
        if cfg["lds"] and knob not in cfg:
            out = [{**c, knob: v} for c in out for v in (False, True)]
    return out


def _warm_from(dirs, p: Problem):
    """Earlier winners as warm starts: this problem's own first, then those of
    problems with the same heads and head dim, then the rest (the space
    repairs whatever does not fit this H, D)."""
    tag = problem_tag(p)
    rows = [r for r in _load_summaries(dirs) if r.get("best_config")]

    def rank(r):
        q = r["problem"]
        same_hd = (q["num_heads"], q["head_dim"]) == (p.num_heads, p.head_dim)
        return (r["tag"] != tag, not same_hd)

    rows.sort(key=rank)
    seen, warm = set(), []
    for r in rows:
        for cfg in _expand_new_knobs(r["best_config"]):
            key = json.dumps(cfg, sort_keys=True)
            if key not in seen:
                seen.add(key)
                warm.append(WarmStart(cfg, source=f"neighbor:{r['tag']}"))
    return warm


def _auto_split_warm(warm):
    """Warm starts with their fixed split replaced by the policy, deduplicated
    (twins that differed only in ``num_splits`` collapse to one)."""
    seen, out = set(), []
    for w in warm:
        cfg = {k: v for k, v in w.config.items() if k != "num_splits"}
        cfg["split"] = "auto"
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(WarmStart(cfg, source=w.source))
    return out


def evaluate_problem(p: Problem, candidates, opts) -> dict:
    """Time named configs on one problem, interleaved, next to the auto pick
    and the reference kernel. A config that fails the builder, spills or
    fails the numeric check is recorded as such and not timed."""
    from aiter.ops.flydsl.fp8_mqa_logits_kernels import _auto_variant

    backend = Fp8MqaLogitsBackend(p, replays=opts.replays)
    space = make_space(p.num_heads, p.head_dim, opts.auto_split)
    auto_tag = _auto_variant(p.s_q, p.s_k, p.num_heads, p.head_dim)
    named = [("auto", config_from_tag(auto_tag)), *candidates]
    items, status = [], {}
    for name, cfg in named:
        if space.key(space.encode(cfg)) is None:
            status[name] = "invalid"
            continue
        comp = backend.compile(cfg, name)
        if not comp.ok:
            status[name] = f"compile: {comp.detail}"
            continue
        spills = {k: v for k, v in comp.resources.items() if k in _SPILL_KEYS and v}
        if spills:
            status[name] = f"spill {spills}"
            continue
        ok, why = backend.check(cfg, name, comp.artifact)
        if not ok:
            status[name] = f"check: {why}"
            continue
        status[name] = "ok"
        items.append((cfg, name, comp.artifact))
    got = backend.measure_batch(items, Fidelity("eval", reps=opts.reps, warmup=1))
    times = {k: statistics.median(v) for (_c, k, _a), v in zip(items, got)}
    out = {
        "problem": p._asdict(),
        "tag": problem_tag(p),
        "auto_variant": auto_tag,
        "reference_us": backend.reference_us(),
        "us": times,
        "status": status,
    }
    out_dir = Path(opts.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{problem_tag(p)}.eval.json").write_text(json.dumps(out, indent=2))
    return out


def _print_summary(rows):
    print(
        f"{'problem':44s} {'auto us':>8s} {'best us':>8s} {'ref us':>8s} "
        f"{'auto/ref':>8s} {'best/ref':>8s}  best"
    )
    for s in rows:
        f = lambda v: f"{v:8.2f}" if isinstance(v, (int, float)) else f"{'-':>8s}"
        print(
            f"{s['tag']:44s} {f(s['auto_us'])} {f(s['best_us'])} "
            f"{f(s['reference_us'])} {f(s['auto_vs_reference'])} "
            f"{f(s['speedup_vs_reference'])}  {s['best']}"
        )


def _fan_out(problems, opts, argv):
    """One worker process per problem, at most one per GPU at a time."""
    gpus = queue.Queue()
    for g in opts.gpus.split(","):
        gpus.put(g.strip())
    base = [a for a in argv if a not in ("--gpus", opts.gpus)]
    # Drop the problem list; each worker gets one.
    cleaned, skip = [], False
    for a in base:
        if a in ("-p", "--problems"):
            skip = True
            continue
        if skip and not a.startswith("-"):
            continue
        skip = False
        cleaned.append(a)
    log_dir = Path(opts.out_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    def work(p):
        g = gpus.get()
        try:
            env = {**os.environ, "HIP_VISIBLE_DEVICES": g}
            cmd = [sys.executable, os.path.abspath(__file__), *cleaned]
            cmd += [
                "-p",
                ",".join(str(int(x) if isinstance(x, bool) else x) for x in p),
            ]
            log = log_dir / f"{problem_tag(p)}.log"
            print(f"[gpu {g}] {problem_tag(p)}", flush=True)
            with open(log, "w") as fh:
                rc = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
            print(f"[gpu {g}] {problem_tag(p)} rc={rc}", flush=True)
        finally:
            gpus.put(g)

    threads = [threading.Thread(target=work, args=(p,)) for p in problems]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = []
    for p in problems:
        if opts.evaluate:
            continue
        f = log_dir / f"{problem_tag(p)}.json"
        if f.exists():
            rows.append(json.loads(f.read_text()))
    if not opts.evaluate:
        _print_summary(rows)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "-p",
        "--problems",
        nargs="+",
        required=True,
        help="s_q,s_k,num_heads,head_dim,clean_logits(0/1),window",
    )
    ap.add_argument("--strategy", default="hill")
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--reps", type=int, default=10, help="samples per evaluation")
    ap.add_argument("--replays", type=int, default=25, help="graph replays per sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--warm", nargs="*", default=[], help="registered variant tags to warm-start"
    )
    ap.add_argument(
        "--warm-from",
        nargs="*",
        default=[],
        help="result directories whose winners warm-start the search",
    )
    ap.add_argument(
        "--evaluate",
        default=None,
        help="no search: time the configs of this JSON list of\n"
        '{"name": ..., "config": {...}} on every problem',
    )
    ap.add_argument("--gpus", default=None, help="fan problems out over these GPUs")
    ap.add_argument(
        "--auto-split",
        action="store_true",
        help="pin the KV split to the launcher's policy (no fixed num_splits)",
    )
    ap.add_argument("--out-dir", default=_DEFAULT_OUT)
    ap.add_argument(
        "--cache-dir", default=None, help="evaluator caches (default: --out-dir)"
    )
    opts = ap.parse_args(argv)
    problems = [parse_problem(s) for s in opts.problems]
    if opts.gpus:
        _fan_out(problems, opts, argv)
        return
    torch.set_default_device("cuda")
    if opts.evaluate:
        cands = json.loads(Path(opts.evaluate).read_text())
        for p in problems:
            evaluate_problem(p, [(c["name"], c["config"]) for c in cands], opts)
        return
    rows = [tune_problem(p, opts) for p in problems]
    _print_summary(rows)


if __name__ == "__main__":
    main()
