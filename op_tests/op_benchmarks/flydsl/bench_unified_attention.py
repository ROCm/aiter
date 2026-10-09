# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A/B benchmark: FlyDSL gfx942 fp8 unified attention against Triton.

Times the FlyDSL kernel directly (past the dispatch cede rule; never the router
or Triton fallback) against aiter's Triton ``unified_attention`` on Gemma-4's
two layer shapes. Triton reads this tree's table, or a ``DEFAULT.json`` given
with ``--triton-config`` (e.g. ROCm/aiter #5650's). That flag also times this
tree's table and reports it as ``triton``. The gate stays on the given table.

Suites: ``acceptance`` (default, the PR evidence) runs prefill at 1K-32K,
decode at context 32K with batch 1/16/64/128, a q256 prefix chunk at 16K and
two mixed prompt-or-chunk + decode steps, on random and sink-like data;
``stress`` is a broad matrix; ``quick`` a short random-data check. --isl,
--prefix-query, --prefix-ctx, --batch and --ctx replace a suite's lists.
Physical pages default to 32 and 64; ``--block-size`` is an optional
runtime-layout check outside the page geomean. K/V are views of one vLLM
cache; softmax scale is 1.0.

Timing: inputs rotate over ~320 MiB of attended bytes (at most 64 sets). Each
candidate is captured once into a HIP graph; replays alternate, each sample at
least 2 ms, and the median per-call time is reported, without host launch
overhead (as in vLLM's graph-captured decode).

Gates per page: FlyDSL agrees with Triton within --tolerance (NaN fails) and
FlyDSL / Triton time is within --margin, in every cell; --enforce-bar exits
nonzero on a miss. The independent fp32 oracle is
op_tests/test_unified_attention.py; this does not replace model E2E.

Output: markdown on stdout (a table per page, rows as they finish, then step
geomeans and gates) and a CSV of every cell (-o, default under aiter_logs/).

Usage, from the repo root (``-m`` puts it on sys.path for ``op_tests``):

    # Acceptance grid against #5650's table, failing on any FlyDSL miss
    python -m op_tests.op_benchmarks.flydsl.bench_unified_attention \\
        --triton-config pr5650/DEFAULT.json --enforce-bar | tee report.md

    # Page 64 decode only, against this tree's table
    python -m op_tests.op_benchmarks.flydsl.bench_unified_attention \\
        --page 64 --cases decode
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import itertools
import math
import platform
import statistics
import subprocess
import time
from functools import partial
from pathlib import Path
from unittest import mock

import flydsl
import torch
import triton

import aiter
import aiter.ops.triton.utils.unified_attention_utils as ua_configs
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.unified_attention_kernels import _launch, is_flydsl_available
from aiter.ops.triton.attention.unified_attention import (
    unified_attention as triton_unified_attention,
)
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir
from op_tests.test_unified_attention import SHAPES_BY_TP

FP8 = torch.float8_e4m3fnuz
# N(0, 1) values are scaled into FNUZ range (max 240); the descale undoes it.
FP8_SCALE = 48.0
ROTATE_BYTES = 320 << 20
MAX_ROTATIONS = 64
# Each timing sample replays a candidate's graph for at least this long.
SAMPLE_US = 2000
SAMPLES = 9
# Gemma-4 31B: 10 full layers and 50 sliding layers per step.
MODEL_LAYERS = {"full": 10, "sliding": 50}
MODEL_ID = "RedHatAI/gemma-4-31B-it-FP8-block"
TARGET_SHAPES_BY_TP = {
    1: {"full": (512, 32, 4, None), "sliding": (256, 32, 16, 1024)},
    4: {"full": (512, 8, 1, None), "sliding": (256, 8, 4, 1024)},
}
for _tp, _shapes in TARGET_SHAPES_BY_TP.items():
    assert SHAPES_BY_TP[_tp] == _shapes, "op test no longer matches Gemma-4-31B"

# Per suite: prefill ISLs; prefix-chunk queries and contexts; decode batches
# and contexts; mixed steps as (queries, chunk context or None for a fresh
# prompt, decodes, decode context); default data patterns.
SUITES = {
    "acceptance": {
        "cases": ["prefill", "prefix", "decode", "mixed"],
        "isl": [1024, 4096, 16384, 32768],
        "prefix_query": [256],
        "prefix_ctx": [16384],
        "batch": [1, 16, 64, 128],
        "ctx": [32768],
        "mixed": [
            (1024, None, 32, 16384),
            (256, 16384, 32, 16384),
        ],
        "data": ["random", "sink"],
    },
    "stress": {
        "cases": ["prefill", "prefix", "decode", "mixed"],
        "isl": [32, 256, 384, 512, 1024, 4096, 8192, 16384, 32768],
        "prefix_query": [16, 64, 256, 512, 2048],
        "prefix_ctx": [4096, 16384, 32768],
        "batch": [1, 4, 16, 32, 64, 128],
        "ctx": [1024, 4096, 32768],
        "mixed": [
            (128, None, 32, 4096),
            (1024, None, 32, 4096),
            (1024, None, 32, 16384),
            (4096, None, 64, 4096),
            (4096, None, 64, 16384),
            (16, 4096, 32, 4096),
            (256, 16384, 32, 4096),
            (2048, 16384, 64, 16384),
        ],
        "data": ["random", "sink"],
    },
    "quick": {
        "cases": ["prefill", "decode"],
        "isl": [1024, 4096, 8192, 16384, 32768],
        "prefix_query": [],
        "prefix_ctx": [],
        "batch": [1, 4, 16, 64],
        "ctx": [1024, 4096, 32768],
        "mixed": [],
        "data": ["random"],
    },
}
GRID_LISTS = ("isl", "prefix_query", "prefix_ctx", "batch", "ctx")
KINDS = ("prefill", "prefix", "decode", "mixed")


def grid(args):
    """(kind, label, [(q_len, k_len), ...]); query i of a sequence sits at
    position k_len - q_len + i."""
    out = []
    if "prefill" in args.cases:
        out += [("prefill", f"ISL {n}", [(n, n)]) for n in args.isl]
    if "prefix" in args.cases:
        out += [
            ("prefix", f"prefix q{q} ctx{k}", [(q, k)])
            for q in args.prefix_query
            for k in args.prefix_ctx
        ]
    if "decode" in args.cases:
        out += [
            ("decode", f"batch {b} ctx {c}", [(1, c)] * b)
            for b in args.batch
            for c in args.ctx
        ]
    if "mixed" in args.cases:
        for q, chunk_ctx, n, c in SUITES[args.suite]["mixed"]:
            head = (
                f"{q} prompt"
                if chunk_ctx is None
                else f"{q}-query chunk at ctx {chunk_ctx}"
            )
            out.append(
                (
                    "mixed",
                    f"{head} + {n} decodes ctx {c}",
                    [(q, chunk_ctx or q)] + [(1, c)] * n,
                )
            )
    return out


def fp8_randn(*shape):
    out = torch.empty(shape, device="cuda", dtype=FP8)
    # Fill in chunks so the bf16 temporary stays small for multi-GiB caches.
    for i in range(0, shape[0], 1024):
        rows = min(1024, shape[0] - i)
        values = torch.randn((rows,) + shape[1:], device="cuda", dtype=torch.bfloat16)
        out[i : i + rows] = (values * FP8_SCALE).clamp(-224, 224).to(FP8)
    return out


def make_set(specs, layer, page, layout, seed, data):
    """One batch: FP8 Q and paged K/V, a shuffled block table, descales."""
    dim, num_heads, num_kv_heads, window = layer
    torch.manual_seed(seed)
    qlens = [q for q, _ in specs]
    klens = [k for _, k in specs]
    pages = [(k + page - 1) // page for k in klens]
    total = sum(pages)
    q = fp8_randn(sum(qlens), num_heads, dim)
    if layout == "vllm":
        # vLLM splits one [blocks, kv_heads, page, 2 * head_dim] cache into views.
        kv = fp8_randn(total, num_kv_heads, page, 2 * dim).transpose(1, 2)
        k, v = kv[..., :dim], kv[..., dim:]
    else:
        k = fp8_randn(total, page, num_kv_heads, dim)
        v = fp8_randn(total, page, num_kv_heads, dim)
    perm = torch.randperm(total, device="cuda", dtype=torch.int32)
    table = torch.zeros(len(specs), max(pages), device="cuda", dtype=torch.int32)
    offset = 0
    for i, n in enumerate(pages):
        table[i, :n] = perm[offset : offset + n]
        offset += n
    if data == "sink":
        # Concentrate probability on page starts: the sharp-softmax regime that
        # random inputs miss, without the sinks API the kernel does not serve.
        q.fill_(FP8_SCALE)
        k.fill_(-FP8_SCALE)
        k[:, 0].fill_(FP8_SCALE)
    keys_read = sum(
        kv if window is None else min(kv, q + window - 1) for q, kv in zip(qlens, klens)
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "table": table,
        "descale": torch.full((1,), 1 / FP8_SCALE, device="cuda"),
        "cu": torch.tensor(
            [0] + list(itertools.accumulate(qlens)), device="cuda", dtype=torch.int32
        ),
        "used": torch.tensor(klens, device="cuda", dtype=torch.int32),
        "max_q": max(qlens),
        "max_k": max(klens),
        "read_bytes": 2 * keys_read * num_kv_heads * dim,
    }


def run_flydsl(s, out, layer, page):
    _launch(
        s["q"], s["k"], s["v"], out, s["cu"], s["max_q"], s["used"], s["max_k"],
        1.0, layer[3], s["table"], s["descale"], s["descale"], s["descale"],
        num_kv_heads=layer[2], block_size=page, num_seqs=s["used"].numel(),
    )  # fmt: skip


def run_triton(s, out, layer):
    window = (-1, -1) if layer[3] is None else (layer[3] - 1, 0)
    triton_unified_attention(
        s["q"], s["k"], s["v"], out, s["cu"], s["max_q"], s["used"], s["max_k"],
        1.0, True, window, s["table"],
        0.0, s["descale"], s["descale"], s["descale"], backend="triton",
    )  # fmt: skip


def _clear_config_caches():
    ua_configs._get_unified_attention_config_cached.cache_clear()
    load_config_json.cache_clear()


@contextlib.contextmanager
def triton_table(path):
    """Triton's unified-attention lookup reads `path`, a DEFAULT.json, instead
    of this tree's table; None keeps this tree's."""
    if path is None:
        yield
        return
    config_dir = str(Path(path).resolve().parent)
    with mock.patch.object(
        ua_configs, "resolve_config_dir", lambda *args, **kwargs: config_dir
    ):
        _clear_config_caches()
        try:
            yield
        finally:
            _clear_config_caches()


def tree_triton_json():
    """This tree's unified-attention DEFAULT.json for the running arch."""
    cfg_dir = resolve_config_dir("attention", ua_configs._CONFIG_NAME, backend="triton")
    return str(Path(cfg_dir) / "DEFAULT.json")


def triton_baselines(config_path):
    """``(label, DEFAULT.json or None)``. None keeps this tree's table.

    The first entry is the gate. When ``--triton-config`` points at another
    file, a second entry times this tree's table as ``triton``.
    """
    if config_path is None:
        return [("tree", None)]
    name = Path(config_path).resolve().parent.name
    baselines = [(name, config_path)]
    same = Path(config_path).resolve() == Path(tree_triton_json()).resolve()
    if not same:
        baselines.append(("tree" if name == "triton" else "triton", None))
    return baselines


def capture(fn, sets, iters):
    """A HIP graph of `iters` rotated calls, after warm-up on a side stream."""
    for i in range(3):
        fn(sets[i % len(sets)])
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for i in range(3):
            fn(sets[i % len(sets)])
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(iters):
            fn(sets[i % len(sets)])
    graph.replay()
    return graph


def replay_us(graph, iters, replays=1):
    """Per-call time over `replays` back-to-back replays of a graph of `iters` calls."""
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(replays):
        graph.replay()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1e3 / (iters * replays)


def time_round_robin(fns, sets, iters):
    """Median per-call time of each candidate. Samples alternate between the
    candidates, in reversed order every other round, so clock and thermal
    drift spread over all of them instead of biasing one.

    fns maps a name to (context, call); the call is captured inside the
    context, which replays do not rerun.
    """
    graphs, replays = {}, {}
    for name, (context, fn) in fns.items():
        with context():
            graphs[name] = capture(fn, sets, iters)
        once = replay_us(graphs[name], iters) * iters
        replays[name] = max(1, math.ceil(SAMPLE_US / once))
    samples = {name: [] for name in fns}
    order = list(fns)
    for rep in range(SAMPLES):
        for name in order if rep % 2 == 0 else order[::-1]:
            samples[name].append(replay_us(graphs[name], iters, replays[name]))
    return {name: statistics.median(times) for name, times in samples.items()}


def diff_vs(got, want):
    """max |got - want| / max |want|; inf when either holds a NaN or inf."""
    got, want = got.float(), want.float()
    if not (torch.isfinite(got).all() and torch.isfinite(want).all()):
        return math.inf
    return ((got - want).abs().max() / want.abs().max()).item()


def bench_cell(kind, label, specs, shape, layer, page, data, args, baselines):
    """Time FlyDSL and each Triton baseline. ``baselines`` is
    ``(label, config path or None)``; numerical diff is against the first."""
    first = make_set(specs, layer, page, args.layout, 0, data)
    count = max(1, min(MAX_ROTATIONS, -(-ROTATE_BYTES // first["read_bytes"])))
    sets = [first] + [
        make_set(specs, layer, page, args.layout, seed, data)
        for seed in range(1, count)
    ]
    out_f = torch.empty(first["q"].shape, device="cuda", dtype=torch.bfloat16)
    out_t = torch.empty_like(out_f)
    run_flydsl(first, out_f, layer, page)
    with triton_table(baselines[0][1]):
        run_triton(first, out_t, layer)
    torch.cuda.synchronize()
    diff = diff_vs(out_f, out_t)
    fns = {
        "flydsl": (
            contextlib.nullcontext,
            partial(run_flydsl, out=out_f, layer=layer, page=page),
        ),
    }
    for bname, path in baselines:
        fns[bname] = (
            partial(triton_table, path),
            partial(run_triton, out=out_t, layer=layer),
        )
    times = time_round_robin(fns, sets, max(len(sets), 10))
    rotated_mib = count * first["read_bytes"] / (1 << 20)
    del sets, first, out_f, out_t
    torch.cuda.empty_cache()
    row = {
        "shape": shape,
        "kind": kind,
        "case": label,
        "page": page,
        "layout": args.layout,
        "data": data,
        "rotations": count,
        "rotated_mib": rotated_mib,
        "flydsl_us": times["flydsl"],
    }
    for bname, _path in baselines:
        row[f"triton_{bname}_us"] = times[bname]
        row[f"ratio_{bname}"] = times["flydsl"] / times[bname]
    row["max_diff_vs_triton"] = diff
    return row


def _md_line(cells):
    return "| " + " | ".join(cells) + " |"


def _column_names(names):
    """In-tree Triton first, then the extra table: triton, then pr5650."""
    return sorted(names, key=lambda name: name not in ("triton", "tree"))


def md_header(names):
    """Header and alignment lines of the per-case markdown table."""
    names = _column_names(names)
    header = ["Shape", "Kind", "Case", "Data", "Page", "FlyDSL µs"]
    header += [f"{name} µs" for name in names]
    header += [f"FlyDSL vs {name}" for name in names]
    align = [":---", ":---", ":---", ":---"] + ["---:"] * (len(header) - 4)
    return _md_line(header) + "\n|" + "|".join(align) + "|"


def md_row(row, names):
    names = _column_names(names)
    cells = [
        row["shape"],
        row["kind"],
        row["case"],
        row["data"],
        str(row["page"]),
        f"{row['flydsl_us']:.1f}",
    ]
    cells += [f"{row[f'triton_{name}_us']:.1f}" for name in names]
    cells += [f"{1 / row[f'ratio_{name}']:.2f}" for name in names]
    return _md_line(cells)


def kind_geomeans(rows, name):
    """(kind, speedup) for a 10-full + 50-sliding step."""
    if set(MODEL_LAYERS) - {row["shape"] for row in rows}:
        return []
    grouped = {}
    for row in rows:
        key = (row["kind"], row["case"], row["data"])
        grouped.setdefault(key, {})[row["shape"]] = row
    by_kind = {}
    for (kind, _case, _data), slot in grouped.items():
        if set(MODEL_LAYERS) - set(slot):
            continue
        flydsl = sum(n * slot[s]["flydsl_us"] for s, n in MODEL_LAYERS.items())
        trit = sum(n * slot[s][f"triton_{name}_us"] for s, n in MODEL_LAYERS.items())
        by_kind.setdefault(kind, []).append(trit / flydsl)
    return [
        (kind, math.exp(sum(map(math.log, ratios)) / len(ratios)))
        for kind, ratios in by_kind.items()
    ]


def md_geomean(groups, names):
    """One row per kind. Each page gets one speedup column per Triton baseline."""
    by_page = [
        (label, {name: dict(kind_geomeans(rows, name)) for name in names})
        for label, rows in groups
    ]
    kinds = [
        kind
        for kind in KINDS
        if all(kind in means[name] for _, means in by_page for name in names)
    ]
    if not kinds:
        return None
    names = _column_names(names)
    headers = ["Kind"]
    for label, _means in by_page:
        headers += [f"FlyDSL vs {name} (Page={label})" for name in names]
    lines = [
        (
            "Gemma-4 5:1 step: 10 full layers + 50 sliding layers, "
            "geomean speedup per physical page and kind."
        ),
        "",
        _md_line(headers),
        "|:---|" + "---:|" * (len(headers) - 1),
    ]
    means = []
    for kind in kinds:
        cells = [kind]
        for _label, page_means in by_page:
            cells += [f"{page_means[name][kind]:.2f}" for name in names]
        means.append(_md_line(cells))
    return "\n".join(lines + means)


def gate_failures(rows, name, args):
    """(correctness, performance) failures, each a list of message lines."""
    correctness, performance = [], []
    for row in rows:
        where = f"{row['shape']} {row['case']} [{row['data']}] page {row['page']}"
        diff = row["max_diff_vs_triton"]
        # NaN-safe: a NaN diff fails, it does not compare false and pass.
        if not diff <= args.tolerance:
            correctness.append(f"- {where}: diff {diff:.6f} > {args.tolerance}")
        ratio = row[f"ratio_{name}"]
        if not ratio <= args.margin:
            performance.append(f"- {where} vs {name}: {ratio:.6f} > {args.margin:.3f}")
    return correctness, performance


def source_version(module):
    path = Path(module.__file__).resolve()
    tree = subprocess.run(
        ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if not tree:
        return str(path)
    sha = subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(tree), "status", "--porcelain", "--untracked-files=normal"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return f"{tree} @ {sha or 'unknown'}{'-dirty' if dirty else ''}"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(
        prog="bench_unified_attention",
        formatter_class=argparse.RawTextHelpFormatter,
        description="A/B benchmark of the FlyDSL gfx942 fp8 unified attention"
        " against Triton on Gemma-4 layer shapes.",
    )
    parser.add_argument(
        "--triton-config",
        metavar="DEFAULT.json",
        help="Extra Triton table to compare against, besides this tree's,"
        "\ne.g. ROCm/aiter #5650's DEFAULT.json. The gate uses this table.",
    )
    parser.add_argument(
        "--shape", choices=["full", "sliding"], nargs="+", default=["full", "sliding"]
    )
    parser.add_argument(
        "--tp",
        type=int,
        choices=sorted(SHAPES_BY_TP),
        default=1,
        help="tensor-parallel degree: each rank's heads",
    )
    pages = parser.add_mutually_exclusive_group()
    pages.add_argument(
        "--page",
        type=int,
        choices=[32, 64, 128],
        nargs="+",
        default=[32, 64],
        help="physical pages to run instead of vLLM cache blocks",
    )
    pages.add_argument(
        "--block-size",
        type=int,
        choices=[32, 64],
        help="optional vLLM layout check: full page=2*block, sliding page=block",
    )
    parser.add_argument(
        "--layout",
        choices=["vllm", "plain"],
        default="vllm",
        help="K/V as views of one vLLM cache, or two contiguous caches",
    )
    parser.add_argument("--suite", choices=list(SUITES), default="acceptance")
    parser.add_argument(
        "--cases",
        choices=KINDS,
        nargs="+",
        help="replace the suite's workload phases",
    )
    parser.add_argument(
        "--data",
        choices=["random", "sink"],
        nargs="+",
        help="replace the suite's data patterns",
    )
    for name in GRID_LISTS:
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=int,
            nargs="+",
            help=f"replace the suite's {name.replace('_', ' ')} list",
        )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.039,
        help="max |candidate - Triton| / max |Triton| allowed per cell",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="FlyDSL time / Triton time allowed per cell",
    )
    parser.add_argument(
        "--enforce-bar",
        action="store_true",
        help="exit nonzero when FlyDSL correctness or performance misses a gate",
    )
    parser.add_argument(
        "-o", metavar="FILE", help="CSV of every cell (default: under aiter_logs/)"
    )
    args = parser.parse_args()
    if args.enforce_bar and args.triton_config is None:
        parser.error("--enforce-bar requires an explicit --triton-config baseline")
    return args


def main():
    # Parsed before the arch gate so `--help` works everywhere.
    args = parse_args()
    if args.triton_config is not None:
        config_path = Path(args.triton_config).resolve()
        if config_path.name != "DEFAULT.json" or not config_path.is_file():
            raise ValueError(
                "--triton-config must be an existing file named DEFAULT.json"
            )
        args.triton_config = str(config_path)
    if get_gfx() != "gfx942" or not is_flydsl_available(torch.cuda.current_device()):
        aiter.logger.warning(
            "FlyDSL unified attention needs gfx942 and FlyDSL; skipping (%s)", get_gfx()
        )
        return
    suite = SUITES[args.suite]
    custom = (
        args.cases is not None
        or args.data is not None
        or any(getattr(args, key) is not None for key in GRID_LISTS)
    )
    if args.cases is None:
        args.cases = suite["cases"]
    for key in GRID_LISTS:
        if getattr(args, key) is None:
            setattr(args, key, suite[key])
    if args.data is None:
        args.data = suite["data"]
    if args.block_size is not None:
        # Optional runtime-layout check; ticket acceptance uses --page 32 64.
        runs = [(f"vLLM block {args.block_size}", None, args.block_size)]
    else:
        runs = [(str(page), page, None) for page in args.page]
    baselines = triton_baselines(args.triton_config)
    names = [bname for bname, _path in baselines]
    name = names[0]
    layers = SHAPES_BY_TP[args.tp]
    cells = grid(args)
    if not cells:
        raise ValueError("selected suite and --cases produce no benchmark cells")

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print(f"- Target: {MODEL_ID}; TP{args.tp}; softmax scale 1.0; BF16 output")
    print(
        f"- Device: {torch.cuda.get_device_name(device)}; {props.gcnArchName}; "
        f"{props.multi_processor_count} CUs; host {platform.node()}"
    )
    print(
        f"- Runtime: torch {torch.__version__}; HIP {torch.version.hip}; "
        f"Triton {triton.__version__}"
    )
    print(f"- aiter {source_version(aiter)}")
    print(f"- FlyDSL {flydsl.__version__}; {source_version(flydsl)}")
    print("- Triton tables:")
    for bname, path in baselines:
        shown = tree_triton_json() if path is None else path
        origin = " (this tree)" if path is None else ""
        print(f"  - {bname} = {shown}{origin}; sha256 {sha256(shown)}")
    print(
        f"- Pages: {', '.join(label for label, _, _ in runs)}; "
        + f"{args.layout} K/V, {'custom' if custom else args.suite} suite, "
        f"data={','.join(args.data)}. "
        "FlyDSL is forced past the dispatch cede rule; no fallback is used. "
        "Speedup is Triton time / FlyDSL time; above 1 is faster.\n"
    )
    if args.suite == "acceptance" and not custom:
        print(
            "- Scope: focused PR performance across prefill, prefix, decode and "
            "mixed phases. Exhaustive correctness is in "
            "op_tests/test_unified_attention.py; OSL 1K belongs to model E2E. "
            "'sink' is a sharp-softmax data pattern, not the attention-sinks API.\n"
        )
    groups = []
    for label, page, block in runs:
        heading = label if block is not None else f"Page {label}"
        print(f"## {heading}\n", flush=True)
        print(md_header(names), flush=True)
        rows = []
        for data, shape in itertools.product(args.data, args.shape):
            cell_page = page or block * (2 if shape == "full" else 1)
            for kind, case, specs in cells:
                row = bench_cell(
                    kind,
                    case,
                    specs,
                    shape,
                    layers[shape],
                    cell_page,
                    data,
                    args,
                    baselines,
                )
                print(md_row(row, names), flush=True)
                rows.append(row)
        groups.append((label, rows))

    geomean = md_geomean(groups, names)
    if geomean:
        print("\n## Geomean\n\n" + geomean)
    rows = [row for _, page_rows in groups for row in page_rows]
    path = args.o
    if path is None:
        geometry = (
            f"b{args.block_size}"
            if args.block_size is not None
            else "-".join(f"p{page}" for page in args.page)
        )
        shapes = "" if len(args.shape) == 2 else "-" + "-".join(args.shape)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = Path("aiter_logs") / (
            f"bench_unified_attention-{stamp}-{geometry}{shapes}-{args.layout}"
            f"-{args.suite}.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: `{Path(path).resolve()}`")

    print("\n## Failures")
    failed = False
    for label, page_rows in groups:
        correctness, performance = gate_failures(page_rows, name, args)
        failed = failed or bool(correctness or performance)
        print(f"\n### {label}\n\nGates over {len(page_rows)} cells:")
        for title, failures in (
            ("Numerical agreement (FlyDSL vs Triton)", correctness),
            (f"FlyDSL time / Triton time <= {args.margin}", performance),
        ):
            print(
                f"\n{title}: {'PASS' if not failures else f'{len(failures)} failures'}"
            )
            for line in failures:
                print(line)
    if args.enforce_bar and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
