# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""All-to-all benchmark: the FlyDSL schedules against RCCL.

Run through ``bench_comm.py --operation a2a``. The collective is an equal-split
all-to-all -- ``all_to_all_single`` without split sizes: every rank's
``(M, K)`` input is cut into ``TP`` equal chunks along its flattened order,
chunk ``j`` goes to rank ``j``, and output chunk ``i`` comes from rank ``i``.

| column            | what runs                                         | wire | exact |
|-------------------|---------------------------------------------------|------|-------|
| ``rccl``          | PyNccl grouped ``ncclSend``/``ncclRecv`` -- the dispatcher's fallback | payload | yes |
| ``rccl_c10d``     | ``dist.all_to_all_single`` (``ncclAllToAll`` on RCCL) | payload | yes |
| ``fly_mesh``      | FlyDSL mesh, register-direct, peer-major stores   | payload | yes |
| ``fly_mesh_atom`` | the same, atom-major (interleaved) stores         | payload | yes |
| ``fly_ring``      | FlyDSL shifted-pairwise ring, register-direct     | payload | yes |
| ``fly_{mesh,ring}_int4`` | the schedule with the INT4 codec           | ~0.28x  | no  |
| ``fly_{mesh,ring}_int6`` | the schedule with the INT6 codec           | ~0.41x  | no  |
| ``fly_auto``      | ``GroupCoordinator.all_to_all`` with ``AITER_FLY_A2A=1``: the production path | per policy | per policy |

Every rank builds every peer's deterministic input, so the reference is the
exact expected output, built locally. Exact rows must match it bit for bit;
the quantizing rows are graded on SQNR against it and must clear their own
floor. ``busbw`` follows the nccl-tests all-to-all convention, ``algbw * (N-1)
/ N``, on the payload dtype -- what the caller's tensor moved, not what the
wire carried.

``us`` is HIP-graph replay time by default (``--timing graph``), as in the
all-reduce benchmark; ``--timing eager`` includes the host launch path.

Examples::

    # 64 MiB of bf16 per rank at TP8
    python3 op_tests/op_benchmarks/flydsl/bench_comm.py --operation a2a -tp 8 -s 4096,8192

    # FlyDSL ring against RCCL only, across the default sizes, TP 2/4/8
    python3 op_tests/op_benchmarks/flydsl/bench_comm.py --operation a2a \\
        -tp 2 4 8 -c rccl rccl_c10d fly_ring fly_ring_int4
"""

import argparse
import os
import re
from dataclasses import dataclass

import pandas as pd
import torch
import torch.distributed as dist
from bench_comm_common import (
    GRAPH_INNER_DEFAULT,
    SUPPORTED_GFX,
    TIMING_CHOICES,
    TIMING_DEFAULT,
    _bench_graph,
    dtype2str,
    load_shapes_csv,
    logger,
    provenance_lines,
    run_ranks,
    sqnr_db,
    write_csv,
    write_report,
)

from aiter import dtypes
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import run_perftest

try:
    from aiter.ops.flydsl.quick_alltoall import FlyQuickAllToAll

    HAS_FLY_A2A = True
except Exception:  # noqa: BLE001
    FlyQuickAllToAll = None
    HAS_FLY_A2A = False

_FLY_A2A_ENV = "AITER_FLY_A2A"
_FLY_WORLDS = (2, 4, 8)
_FLY_ARCHS = ("gfx942", "gfx950")
_QUANT_CODECS = ("int4", "int6")

# Single quantization per element, so a codec's floor is its own round-trip
# SQNR: INT4 ~19-25 dB, INT6 ~31-37 dB on Gaussian data. ~5 dB of slack.
_SQNR_FLOOR = {"int4": 15.0, "int6": 26.0}
DEFAULT_MIN_SQNR = 15.0


@dataclass(frozen=True)
class Candidate:
    """One thing that can perform the all-to-all, plus how to grade it.

    ``family`` selects the launcher: ``rccl`` (PyNccl), ``rccl_c10d``,
    ``fly`` (a pinned ``FlyQuickAllToAll``) or ``fly_auto`` (the dispatcher).
    For ``fly`` rows, ``None`` knobs leave the engine's own ladder in charge.
    """

    key: str
    family: str
    algorithm: str | None = None
    codec: str | None = None
    order: str = "peer"
    super_tile: int | None = None
    block: int | None = None
    grid_cap: int | None = None

    @property
    def exact(self) -> bool:
        return self.codec not in _QUANT_CODECS

    @property
    def sqnr_floor(self) -> float | None:
        return _SQNR_FLOOR.get(self.codec)

    @property
    def fly_cfg(self) -> tuple:
        """Identity of the ``FlyQuickAllToAll`` engine this candidate needs."""
        return (
            self.algorithm,
            self.codec,
            self.order,
            self.super_tile,
            self.block,
            self.grid_cap,
        )


CANDIDATES = (
    Candidate("rccl", "rccl"),
    Candidate("rccl_c10d", "rccl_c10d"),
    Candidate("fly_mesh", "fly", algorithm="mesh"),
    Candidate("fly_mesh_atom", "fly", algorithm="mesh", order="atom"),
    Candidate("fly_ring", "fly", algorithm="ring"),
    Candidate("fly_mesh_int4", "fly", algorithm="mesh", codec="int4"),
    Candidate("fly_ring_int4", "fly", algorithm="ring", codec="int4"),
    Candidate("fly_mesh_int6", "fly", algorithm="mesh", codec="int6"),
    Candidate("fly_ring_int6", "fly", algorithm="ring", codec="int6"),
    Candidate("fly_auto", "fly_auto"),
)
_BY_KEY = {c.key: c for c in CANDIDATES}
PRIMARY = "rccl"

# 64 MiB of bf16 is the production message; the smaller rows place the
# crossover with RCCL. Hidden 8192: 16 KiB per token.
L_SHAPE = [(m, 8192) for m in (8, 32, 64, 256, 1024, 2048, 4096)]


def _make_input(rank: int, tokens: int, hidden: int, dtype, device) -> torch.Tensor:
    """Rank *rank*'s input, identical whichever rank generates it."""
    gen = torch.Generator(device=device).manual_seed(4321 + rank)
    return torch.randn(
        tokens, hidden, generator=gen, dtype=torch.float32, device=device
    ).to(dtype)


def _reference(rank, tp, tokens, hidden, dtype, device) -> torch.Tensor:
    """What rank *rank* must hold afterwards: chunk *rank* of every rank."""
    chunk = tokens * hidden // tp
    return torch.cat(
        [
            _make_input(src, tokens, hidden, dtype, device).view(-1)[
                rank * chunk : (rank + 1) * chunk
            ]
            for src in range(tp)
        ]
    ).view(tokens, hidden)


def applicable(cand: Candidate, tp: int, dtype, nbytes: int) -> bool:
    if nbytes % (16 * tp):
        return cand.family == "rccl_c10d"
    if cand.family in ("rccl", "rccl_c10d"):
        return True
    if not HAS_FLY_A2A or get_gfx() not in _FLY_ARCHS or tp not in _FLY_WORLDS:
        return False
    return not (cand.codec in _QUANT_CODECS and dtype != dtypes.bf16)


_RANK_FIELD = re.compile(r"_r\d+_")


def _worker(tp, rank, shapes, dtype, keys, init_method, opts):
    """One rank: join the group once, build every engine, sweep every shape."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp, 1)
    tp_group = get_tp_group()
    group = tp_group.device_group
    pynccl = tp_group.device_communicator.pynccl_comm
    if pynccl is not None and pynccl.disabled:
        pynccl = None
    fly_a2a_comm = tp_group.device_communicator.fly_a2a_comm

    cands = [_BY_KEY[k] for k in keys]
    # One engine per distinct configuration, built in a fixed order on every
    # rank: each construction exchanges IPC handles, which is a collective.
    fly = {}
    if HAS_FLY_A2A and get_gfx() in _FLY_ARCHS and tp in _FLY_WORLDS:
        for cfg in sorted(
            {c.fly_cfg for c in cands if c.family == "fly"},
            key=lambda cfg: tuple(str(v) for v in cfg),
        ):
            algorithm, codec, order, super_tile, block, grid_cap = cfg
            if codec in _QUANT_CODECS and dtype != dtypes.bf16:
                continue
            eng = FlyQuickAllToAll(
                group=tp_group.cpu_group,
                device=device,
                rank=rank,
                world_size=tp,
                algorithm=algorithm,
                codec=codec,
                order=order,
                super_tile=super_tile,
                block=block,
                grid_cap=grid_cap,
            )
            eng.preload()
            fly[cfg] = eng

    # Warm RCCL and align ranks before any timing.
    warm = torch.zeros(tp * 16, device=device)
    dist.all_to_all_single(torch.empty_like(warm), warm, group=group)
    torch.cuda.synchronize()

    rows = []
    try:
        for tokens, hidden in shapes:
            x = _make_input(rank, tokens, hidden, dtype, device)
            ref = _reference(rank, tp, tokens, hidden, dtype, device)
            nbytes = x.numel() * x.element_size()
            ret = {"nbytes": nbytes}
            for cand in cands:
                if not applicable(cand, tp, dtype, nbytes):
                    continue
                if cand.family == "fly" and cand.fly_cfg not in fly:
                    continue
                out = torch.empty_like(x)
                variant = None
                if cand.family == "rccl":
                    if pynccl is None:
                        continue

                    def thunk(o=out, src=x):
                        pynccl.all_to_all(o, src)
                        return o

                elif cand.family == "rccl_c10d":

                    def thunk(o=out, src=x):
                        dist.all_to_all_single(o.view(-1), src.view(-1), group=group)
                        return o

                elif cand.family == "fly":
                    eng = fly[cand.fly_cfg]
                    variant = eng.variant(nbytes)

                    def thunk(o=out, src=x, eng=eng):
                        eng.all_to_all(src, o)
                        return o

                else:  # fly_auto
                    routed = (
                        fly_a2a_comm is not None and fly_a2a_comm.should_all_to_all(x)
                    )
                    variant = fly_a2a_comm.variant(nbytes) if routed else "rccl"

                    def thunk(o=out, src=x):
                        return tp_group.all_to_all(src, o)

                dist.barrier(group=group)
                torch.cuda.synchronize()
                label = f"{cand.key} tp{tp} {tokens}x{hidden}"
                if opts["timing"] == "graph":
                    got, us = _bench_graph(
                        thunk,
                        num_iters=opts["iters"],
                        num_warmup=opts["warmup"],
                        inner=opts["graph_inner"],
                        group=group,
                        label=label,
                    )
                else:
                    got, us = run_perftest(
                        thunk,
                        num_iters=opts["iters"],
                        num_warmup=opts["warmup"],
                        use_cuda_event=True,
                    )
                mismatch = int((got.view(torch.int16) != ref.view(torch.int16)).sum())
                sqnr = sqnr_db(got, ref)
                if cand.exact and mismatch:
                    logger.warning(
                        "%s rank%d: %d lanes differ from the exact reference",
                        label,
                        rank,
                        mismatch,
                    )
                if cand.sqnr_floor is not None and sqnr < cand.sqnr_floor:
                    logger.warning(
                        "%s rank%d: SQNR %.2f dB below the %.1f dB floor",
                        label,
                        rank,
                        sqnr,
                        cand.sqnr_floor,
                    )
                ret[f"{cand.key}_us"] = us
                ret[f"{cand.key}_sqnr"] = sqnr
                ret[f"{cand.key}_mismatch"] = mismatch
                ret[f"{cand.key}_variant"] = variant
            rows.append(ret)
            del x, ref
            torch.cuda.empty_cache()
    finally:
        for eng in fly.values():
            eng.close()
        destroy_model_parallel()
        destroy_distributed_environment()
    return rows


def _row(tp, tokens, hidden, dtype, rank_rets, keys) -> dict:
    """Collapse one shape's per-rank scalars: the slowest rank's time, the
    worst rank's accuracy."""
    nbytes = rank_rets[0]["nbytes"]
    row = {
        "gfx": get_gfx(),
        "dtype": dtype2str(dtype),
        "TP": tp,
        "M": tokens,
        "K": hidden,
        "payload size (KiB)": nbytes / 1024,
        "_nbytes": nbytes,
    }
    for key in keys:
        if f"{key}_us" not in rank_rets[0]:
            continue
        us = max(r[f"{key}_us"] for r in rank_rets)
        row[f"{key} us"] = us
        row[f"{key} SQNR dB"] = min(r[f"{key}_sqnr"] for r in rank_rets)
        row[f"{key} mismatch"] = max(r[f"{key}_mismatch"] for r in rank_rets)
        row[f"{key} busbw GB/s"] = nbytes * (tp - 1) / tp / us / 1e3
        variant = rank_rets[0][f"{key}_variant"]
        row[f"{key} variant"] = (
            None if variant is None else _RANK_FIELD.sub("_r*_", variant)
        )
    return row


def run_sweep(tp, shapes, dtype, keys, opts) -> list[dict]:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    logger.info(
        "a2a TP%d %s: %d shape(s), %d iters, timing %s",
        tp,
        dtype2str(dtype),
        len(shapes),
        opts["iters"],
        opts["timing"],
    )
    per_rank = run_ranks(
        tp, _worker, lambda r: (tp, r, shapes, dtype, keys, init_method, opts)
    )
    return [
        _row(tp, tokens, hidden, dtype, [pr[i] for pr in per_rank], keys)
        for i, (tokens, hidden) in enumerate(shapes)
    ]


def _passes(row, key, min_sqnr) -> bool:
    cand = _BY_KEY[key]
    if cand.exact:
        return row.get(f"{key} mismatch") == 0
    return row.get(f"{key} SQNR dB", float("-inf")) >= min_sqnr


def summary_table(df, keys, baseline, min_sqnr) -> pd.DataFrame:
    """Per shape: the baseline, the fastest exact candidate, and the fastest
    one clearing *min_sqnr*, each with its speedup over the baseline."""
    out = []
    for _, row in df.iterrows():
        timed = [k for k in keys if pd.notna(row.get(f"{k} us"))]
        base_us = row.get(f"{baseline} us")
        rec = {
            "TP": row["TP"],
            "M": row["M"],
            "K": row["K"],
            "payload size (KiB)": row["payload size (KiB)"],
            f"{baseline} us": base_us,
        }
        for label, pool in (
            (
                "fastest exact",
                [k for k in timed if _BY_KEY[k].exact and _passes(row, k, min_sqnr)],
            ),
            ("fastest", [k for k in timed if _passes(row, k, min_sqnr)]),
        ):
            best = min(pool, key=lambda k: row[f"{k} us"], default=None)
            rec[label] = best
            rec[f"{label} us"] = None if best is None else row[f"{best} us"]
            rec[f"{label} vs {baseline}"] = (
                None
                if best is None or pd.isna(base_us)
                else base_us / row[f"{best} us"]
            )
            if label == "fastest":
                rec["fastest SQNR dB"] = (
                    None if best is None else row[f"{best} SQNR dB"]
                )
        out.append(rec)
    return pd.DataFrame(out)


def case_tables(df, keys, baseline):
    """One table per shape: every candidate's time, accuracy and bandwidth."""
    for _, row in df.iterrows():
        base_us = row.get(f"{baseline} us")
        rows = []
        for key in keys:
            us = row.get(f"{key} us")
            if us is None or pd.isna(us):
                continue
            rows.append(
                {
                    "candidate": key,
                    "us": us,
                    f"vs {baseline}": None if pd.isna(base_us) else base_us / us,
                    "SQNR dB": row[f"{key} SQNR dB"],
                    "mismatch": row[f"{key} mismatch"],
                    "busbw GB/s": row[f"{key} busbw GB/s"],
                    "variant": row[f"{key} variant"],
                }
            )
        title = (
            f"TP{row['TP']}, {row['dtype']}, M={row['M']}, K={row['K']}, "
            f"{row['payload size (KiB)']:g} KiB"
        )
        yield title, pd.DataFrame(rows)


def main(argv=None):
    if get_gfx() not in SUPPORTED_GFX:
        logger.warning("all-to-all benchmark unsupported on %s; skipping", get_gfx())
        return
    keys_all = [c.key for c in CANDIDATES]
    parser = argparse.ArgumentParser(
        prog="bench_comm.py --operation a2a",
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "-tp",
        "--tp",
        type=int,
        nargs="*",
        choices=[2, 4, 8],
        default=None,
        help="world size(s). Default: TP8 if visible, else the largest that is.",
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        default=L_SHAPE,
        help="(tokens, hidden) per-rank input shapes, e.g. -s 4096,8192",
    )
    parser.add_argument(
        "--shape-csv",
        metavar="PATH",
        default=None,
        help="M,K shapes from a CSV; exclusive with -s.",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        nargs="*",
        default=["bf16"],
        choices=["bf16", "fp16"],
        help="payload dtype(s). The quantizing candidates are bf16-only.",
    )
    parser.add_argument(
        "-c",
        "--candidates",
        nargs="*",
        default=None,
        choices=keys_all,
        help="candidates to run. Default: all.",
    )
    parser.add_argument(
        "-b",
        "--baseline",
        default=PRIMARY,
        choices=keys_all,
        help=f"the column speedups are reported against. Default {PRIMARY}.",
    )
    parser.add_argument(
        "--list-candidates", action="store_true", help="print the candidates and exit"
    )
    parser.add_argument("--iters", type=int, default=101, help="timed iterations")
    parser.add_argument("--warmup", type=int, default=5, help="warmup iterations")
    parser.add_argument(
        "--timing",
        choices=TIMING_CHOICES,
        default=TIMING_DEFAULT,
        help="graph: HIP-graph replay (default); eager: host path included.",
    )
    parser.add_argument(
        "--graph-inner",
        type=int,
        default=GRAPH_INNER_DEFAULT,
        help="collectives per captured graph under --timing graph.",
    )
    parser.add_argument(
        "--min-sqnr",
        type=float,
        default=DEFAULT_MIN_SQNR,
        help="accuracy floor of the summary's `fastest` column, in dB.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="also write the markdown report to PATH.",
    )
    parser.add_argument(
        "--output-csv",
        metavar="PATH",
        default=None,
        help="also write the raw per-shape dataframe to PATH.",
    )
    args = parser.parse_args(argv)

    if args.list_candidates:
        for c in CANDIDATES:
            print(f"  {c.key:<16} {c.family:<10} exact={c.exact}")
        return
    if args.graph_inner < 1:
        parser.error("--graph-inner must be positive")
    if args.shape_csv is not None:
        if args.shape is not L_SHAPE:
            parser.error("--shape-csv and -s/--shape are mutually exclusive")
        args.shape = load_shapes_csv(args.shape_csv)
    visible = torch.cuda.device_count()
    tps = args.tp or [max(t for t in (2, 4, 8) if t <= visible)]
    tps = [t for t in tps if t <= visible]
    if not tps:
        logger.warning("no requested TP size fits %d visible GPUs; skipping", visible)
        return
    keys = args.candidates or keys_all
    if args.baseline not in keys:
        keys = [args.baseline, *keys]
    if "fly_auto" in keys:
        # The dispatcher is opt-in; set before the ranks spawn so they inherit it.
        os.environ[_FLY_A2A_ENV] = "1"
    opts = {
        "iters": args.iters,
        "warmup": args.warmup,
        "timing": args.timing,
        "graph_inner": args.graph_inner,
    }

    sections = []
    for dtype_name in args.dtype:
        dtype = dtypes.d_dtypes[dtype_name]
        rows = []
        for tp in tps:
            rows += run_sweep(tp, args.shape, dtype, keys, opts)
        df = pd.DataFrame(rows)

        title = f"{dtype_name} all-to-all summary"
        md = summary_table(df, keys, args.baseline, args.min_sqnr).to_markdown(
            index=False, floatfmt=".4g", missingval="n/a"
        )
        logger.info("%s (markdown):\n%s", title, md)
        sections.append((title, md))
        case_md = "\n\n".join(
            f"### {t}\n\n"
            + cdf.to_markdown(index=False, floatfmt=".4g", missingval="n/a")
            for t, cdf in case_tables(df, keys, args.baseline)
        )
        title = f"{dtype_name} all-to-all latency & accuracy by case"
        logger.info("%s (markdown):\n%s", title, case_md)
        sections.append((title, case_md))
        if args.output_csv:
            write_csv(args.output_csv, df, dtype_name, len(args.dtype) > 1)

    if args.output:
        header = provenance_lines("aiter all-to-all benchmark", args, visible)
        header += [
            f"- FlyDSL all-to-all available: {HAS_FLY_A2A}",
            f"- summary `fastest` accuracy floor: {args.min_sqnr} dB",
            "",
            "Exact candidates must reproduce the reference bit for bit (`mismatch`",
            "is the count of 16-bit lanes that differ); the quantizing ones are",
            "graded on `SQNR dB`. `busbw` is `bytes * (N-1)/N / us` on the payload.",
            "",
        ]
        write_report(args.output, header, sections)
