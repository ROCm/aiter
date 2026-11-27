# SPDX-License-Identifier: MIT

r"""Sweep RDNA FlyDSL MXFP4 W4A16 configs for a list of M values at fixed N and K.

    python -m op_tests.tuners.tune_mxfp4_w4a16 --m 1-5 --n 34816 --k 5120 \
        --csv quark-gate-up-trials.csv

The default search crosses every launch knob. Tile M/N and stage K use
16, 32, 64, and 128. Traversal covers every
distinct group width for the shape. Use --space to replace any axis (for
example, --space stage_k=16-256:16). This can be a long sweep. Use --max-trials
to run it in batches and reuse --csv to resume. --continue-from imports a
previous CSV when continuing into a different output file. Progress requires
tqdm in the tuning environment. --output accepts a selector JSON path named
GEMM-MXFP4-W4A16-N=<N>-K=<K>.json.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import statistics
from contextlib import nullcontext
from dataclasses import asdict, fields, replace
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile

import flydsl
import torch
from flydsl.compiler.diagnostics import DSLCompileError
from flydsl.compiler.external_llvm import ExternalLLVMError
from tqdm import tqdm

from aiter.jit.utils.chip_info import get_cu_num_device, get_gfx_runtime_for_device
from aiter.ops.flydsl import gemm_mxfp4_w4a16 as gemm
from aiter.ops.flydsl.gemm_mxfp4_w4a16_config import (
    SUPPORTED_ARCHS,
    MXFP4W4A16Config,
    validate_config,
)
from aiter.ops.flydsl.kernels.gemm.mxfp4_w4a16.rdna3 import mxfp4_w4a16 as kernel

_GEOMETRY = {
    "group_tile_m": (16, 32, 64, 128),
    "group_tile_n": (16, 32, 64, 128),
    "stage_k": (16, 32, 64, 128),
    "waves_per_group_m": (1, 2, 4, 8),
    "waves_per_group_n": (1, 2, 4, 8),
    "wave_groups_m": (1, 2, 4),
    "wave_groups_n": (1, 2, 4),
    "left_in_registers": (True, False),
    "right_in_registers": (False, True),
}
_SCHEDULE = {
    "split_k": (1, 2, 4),
    "k_unroll": (1, 2, 4, 8, 16),
    # RDNA3.5 ISA §4.1.1. Use SLC for GL2 streaming and test DLC's MALL
    # no-allocate hint on the larger B stream. Inputs are read-only, so the
    # default sweep leaves GLC clear to permit first-level cache hits.
    "left_cache_modifier": (0, 2),
    "right_cache_modifier": (0, 2, 4, 6),
}
_TRAVERSAL = "traversal_group_m"
_BOOL_FIELDS = {f.name for f in fields(MXFP4W4A16Config) if f.type is bool}
_CONFIG_FIELDS = [f.name for f in fields(MXFP4W4A16Config)]
_RESULT_FIELDS = ["status", "median_us", "detail"]
_TRIAL_STATUSES = ("ok", "compile_error", "invalid", "incorrect", "out_of_memory")


def _integers(text):
    """Parse comma-separated integers and inclusive ranges with optional steps."""
    values = []
    for item in text.split(","):
        first, dash, last = item.strip().partition("-")
        bound, colon, step_text = last.partition(":")
        start = int(first)
        stop = int(bound) if dash else start
        step = int(step_text) if colon else 1
        if start < 0 or stop < start or step <= 0:
            raise ValueError(f"Invalid integer range {item!r}")
        values.extend(range(start, stop + 1, step))
    return tuple(dict.fromkeys(values))


def _parse_space(overrides):
    axes = {}
    for override in overrides:
        name, sep, text = override.partition("=")
        if not sep or name not in _CONFIG_FIELDS or name in axes:
            raise ValueError(f"Expected a unique CONFIG_FIELD=VALUES, got {override!r}")
        values = _integers(text)
        if name in _BOOL_FIELDS:
            if any(value not in (0, 1) for value in values):
                raise ValueError(f"{name} takes 0 or 1")
            values = tuple(bool(value) for value in values)
        elif name in ("group_tile_m", "group_tile_n", "stage_k"):
            if any(value < 16 or value % 16 for value in values):
                raise ValueError(f"{name} requires positive multiples of 16")
        elif name == _TRAVERSAL:
            if 0 in values:
                raise ValueError(f"{name} must be positive")
        elif name in ("left_cache_modifier", "right_cache_modifier"):
            if any(value not in range(8) for value in values):
                raise ValueError(f"{name} requires a three-bit gfx11 modifier")
        elif any(value not in (_GEOMETRY | _SCHEDULE)[name] for value in values):
            raise ValueError(f"Unsupported values for {name}: {values}")
        axes[name] = values
    return axes


def _traversal_widths(m, config, override):
    block_m = config.group_tile_m * config.wave_groups_m
    blocks_m = (m + block_m - 1) // block_m
    # The kernel clamps larger traversal widths to blocks_m.
    return override if override is not None else range(1, blocks_m + 1)


def _legal_geometries(shape, geometry, traversal):
    for values in product(*geometry.values()):
        config = MXFP4W4A16Config(**dict(zip(geometry, values)))
        try:
            validate_config(config, shape)
        except ValueError:
            continue
        yield config, _traversal_widths(shape[0], config, traversal)


def _configs(geometries, schedule):
    for config, widths in geometries:
        for width, *knobs in product(widths, *schedule.values()):
            yield replace(config, traversal_group_m=width, **dict(zip(schedule, knobs)))


def _identity(device, n, k, args):
    source = hashlib.sha256()
    for module in (gemm, kernel):
        source.update(Path(module.__file__).read_bytes())
    return {
        "schema": "1",
        "gfx": get_gfx_runtime_for_device(device),
        "cu_count": str(get_cu_num_device(device.index)),
        "device": torch.cuda.get_device_name(device),
        "wgp_mode": os.getenv("GPU_ENABLE_WGP_MODE", "1"),
        "torch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "flydsl_version": flydsl.__version__,
        "kernel_sha256": source.hexdigest(),
        "batches": str(args.batches),
        "replays": str(args.replays),
        "graph_gemms": str(args.graph_gemms),
        "n": str(n),
        "k": str(k),
    }


def _read_csv(handle, columns, identity, history, *, repair_tail=False, copy_to=None):
    """Restore matching trials and optionally copy new rows into a checkpoint."""
    handle.seek(0)
    offset = 0
    line_number = 0
    loaded = 0
    while line := handle.readline():
        if not line.endswith("\n"):
            if repair_tail:
                handle.seek(offset)
                handle.truncate()
            break
        offset = handle.tell()
        line_number += 1
        try:
            values = next(csv.reader([line], strict=True))
            if line_number == 1:
                if values != columns:
                    raise ValueError("Unexpected MXFP4 tuning CSV header")
                continue
            if len(values) != len(columns):
                raise ValueError("Incorrect CSV column count")
            row = dict(zip(columns, values))
            if any(row[key] != value for key, value in identity.items()):
                continue
            m = int(row["m"])
            if m not in history:
                continue
            if any(row[name] not in ("True", "False") for name in _BOOL_FIELDS):
                raise ValueError("Invalid boolean config field")
            config = MXFP4W4A16Config(
                **{
                    name: (
                        row[name] == "True" if name in _BOOL_FIELDS else int(row[name])
                    )
                    for name in _CONFIG_FIELDS
                }
            )
            status = row["status"]
            latency = float(row["median_us"]) if status == "ok" else None
            if status not in _TRIAL_STATUSES or (
                latency is not None and (not math.isfinite(latency) or latency <= 0)
            ):
                raise ValueError("Invalid trial result")
            if copy_to is not None:
                if config in history[m]:
                    continue
                copy_to.writerow(row)
            history[m][config] = latency
            loaded += 1
        except (ValueError, csv.Error) as error:
            raise ValueError(
                f"Invalid tuning CSV {handle.name} line {line_number}: {error}"
            ) from error
    return line_number > 0, loaded


def _make_weights(n, k, device):
    rng = torch.Generator(device=device).manual_seed(2026)
    packed = torch.randint(
        256, (n, k // 2), device=device, dtype=torch.uint8, generator=rng
    )
    scales = torch.randint(
        121, 128, (n, (k + 31) // 32), device=device, dtype=torch.uint8, generator=rng
    )
    return packed, scales, gemm.prepare_gemm_mxfp4_w4a16_weight(packed, scales)


def _make_problem(shape, device, packed, scales):
    m, n, k = shape
    rng = torch.Generator(device=device).manual_seed(289)
    a = (torch.randn((m, k), device=device, generator=rng) * k**-0.5).bfloat16()
    expected = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=device)
    a_fp32 = a.float()
    # Decode a chunk at a time to bound the FP32 reference's memory use.
    for start in range(0, n, 256):
        words = packed[start : start + 256]
        codes = torch.stack((words & 15, words >> 4), dim=-1).reshape(-1, k)
        weights = levels[(codes & 7).long()]
        weights = torch.where(codes & 8 != 0, -weights, weights)
        scale = (scales[start : start + 256].float() - 127).exp2()
        weights *= scale.repeat_interleave(32, dim=1)[:, :k]
        expected[:, start : start + 256] = a_fp32 @ weights.T
    return a, torch.empty_like(expected), expected


def _measure(a, prepared, out, expected, config, args):
    def run():
        gemm.flydsl_gemm_mxfp4_w4a16(a, prepared, out=out, config=config)

    out.fill_(float("nan"))
    run()
    torch.testing.assert_close(out, expected, rtol=0.016, atol=0.03125)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(args.graph_gemms):
            run()
    out.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(out, expected, rtol=0.016, atol=0.03125)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(args.batches):
        start.record()
        for _ in range(args.replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(
            start.elapsed_time(end) * 1000 / (args.replays * args.graph_gemms)
        )
    return statistics.median(samples)


def _trial(shape, problem, prepared, config, args):
    a, out, expected = problem
    try:
        latency = _measure(a, prepared, out, expected, config, args)
        if not math.isfinite(latency) or latency <= 0:
            raise RuntimeError(f"Invalid measured latency: {latency}")
        return "ok", latency, ""
    except (DSLCompileError, ExternalLLVMError) as error:
        return "compile_error", None, " ".join(str(error).split())[:500]
    except ValueError as error:
        return "invalid", None, " ".join(str(error).split())[:500]
    except AssertionError as error:
        return "incorrect", None, " ".join(str(error).split())[:500]
    except torch.OutOfMemoryError as error:
        return "out_of_memory", None, " ".join(str(error).split())[:500]
    finally:
        # Each config is used once per shape. Release its loaded code and IR
        # after the graph is done to bound memory during long sweeps.
        launch = kernel.launch_mxfp4_w4a16
        getattr(launch, "_cf_by_specialization", {}).pop(
            (a.device, shape, config), None
        )
        launch._mem_cache.clear()
        launch._last_compiled = None
        gemm._eager_workspace.cache_clear()


def _restored_count(history, geometries, schedule):
    if not history:
        return 0
    widths_by_geometry = {
        tuple(getattr(config, name) for name in _GEOMETRY): widths
        for config, widths in geometries
    }
    return sum(
        config.traversal_group_m
        in widths_by_geometry.get(
            tuple(getattr(config, name) for name in _GEOMETRY), ()
        )
        and all(getattr(config, name) in values for name, values in schedule.items())
        for config in history
    )


def _sweep(args, device, identity, writer, handle, history):
    winners = {}
    weights = None
    new_trials = 0
    failed = False
    geometry = _GEOMETRY | {key: args.axes[key] for key in _GEOMETRY.keys() & args.axes}
    schedule = _SCHEDULE | {key: args.axes[key] for key in _SCHEDULE.keys() & args.axes}
    schedules_per_width = math.prod(map(len, schedule.values()))
    for m in args.m:
        shape = (m, args.n, args.k)
        geometries = list(_legal_geometries(shape, geometry, args.axes.get(_TRAVERSAL)))
        total = schedules_per_width * sum(len(widths) for _, widths in geometries)
        restored = _restored_count(history[m], geometries, schedule)
        problem = None
        best = None
        tested = skipped = rejected = 0
        last_error = ""
        complete = True
        with tqdm(
            total=total,
            initial=restored,
            desc=f"M={m} N={args.n} K={args.k}",
            unit="cfg",
            dynamic_ncols=True,
            postfix=f"resumed={restored:,}",
        ) as progress:
            for config in _configs(geometries, schedule):
                if config in history[m]:
                    latency = history[m][config]
                    skipped += 1
                else:
                    if args.max_trials is not None and new_trials >= args.max_trials:
                        complete = False
                        break
                    if weights is None:
                        weights = _make_weights(args.n, args.k, device)
                    if problem is None:
                        problem = _make_problem(shape, device, *weights[:2])
                    status, latency, detail = _trial(
                        shape, problem, weights[2], config, args
                    )
                    if detail:
                        last_error = detail
                    if writer is not None:
                        writer.writerow(
                            {
                                **identity,
                                "m": m,
                                **asdict(config),
                                "status": status,
                                "median_us": latency,
                                "detail": detail,
                            }
                        )
                        handle.flush()
                    tested += 1
                    new_trials += 1
                    progress.update()
                if latency is None:
                    rejected += 1
                elif best is None or latency < best[0]:
                    best = (latency, config)

        print(
            f"{shape} tested {tested}, skipped {skipped}, rejected {rejected}",
            flush=True,
        )
        if not complete:
            if best is not None:
                latency, config = best
                print(
                    f"{shape} best so far {latency:.3f} us {json.dumps(asdict(config))}"
                )
            print(f"Stopped at {args.max_trials} new trials while tuning {shape}.")
            break
        if best is None:
            print(f"{shape} has no verified config. {last_error}", flush=True)
            failed = True
            continue
        latency, config = best
        winners[f"M_EQ_{m}"] = asdict(config)
        print(f"{shape} best {latency:.3f} us {json.dumps(asdict(config))}", flush=True)
    return winners, failed


def _write_winners(path, winners):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    rows.update(winners)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as tmp:
        json.dump(rows, tmp, indent=2)
        tmp.write("\n")
    Path(tmp.name).replace(path)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Search axes match the config fields\n  " + "\n  ".join(_CONFIG_FIELDS),
    )
    parser.add_argument(
        "--m", nargs="+", required=True, help="M values, e.g. 1,2,3,4,5 or 1-5"
    )
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument(
        "--device", type=int, default=0, help="Visible ROCm device index"
    )
    parser.add_argument(
        "--csv", type=Path, help="Append trials and resume from this CSV"
    )
    parser.add_argument(
        "--continue-from",
        type=Path,
        metavar="PREVIOUS_CSV",
        help="Resume this CSV in place or import matching rows into --csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Merge completed M winners into a selector JSON file",
    )
    parser.add_argument(
        "--space",
        action="append",
        default=[],
        metavar="FIELD=VALUES",
        help="Override one search axis (repeatable, booleans use 0/1)",
    )
    parser.add_argument(
        "--max-trials", type=int, help="Limit new trials per invocation"
    )
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--graph-gemms", type=int, default=4)
    args = parser.parse_args()
    try:
        args.m = _integers(",".join(args.m))
        args.axes = _parse_space(args.space)
    except ValueError as error:
        parser.error(str(error))
    if min(*args.m, args.n, args.k) <= 0 or args.k % 2:
        parser.error("M, N, K must be positive and K must be even")
    if (
        max(2 * max(args.m) * args.k, 2 * max(args.m) * args.n, args.n * args.k)
        >= 1 << 31
    ):
        parser.error("GEMM buffer offsets must fit in signed 32 bits")
    if min(args.batches, args.replays, args.graph_gemms) <= 0 or (
        args.max_trials is not None and args.max_trials <= 0
    ):
        parser.error("Timing and trial counts must be positive")
    if args.continue_from is not None:
        if not args.continue_from.is_file():
            parser.error(f"Previous tuning CSV does not exist: {args.continue_from}")
        if args.csv is None:
            args.csv = args.continue_from
        if (
            args.output
            and args.output.exists()
            and args.output.samefile(args.continue_from)
        ):
            parser.error("Previous CSV and JSON output paths must differ")
    if args.csv and args.output and args.csv.resolve() == args.output.resolve():
        parser.error("CSV and JSON output paths must differ")
    if torch.version.hip is None or not 0 <= args.device < torch.cuda.device_count():
        parser.error("Select a visible ROCm GPU with --device")
    return args


@torch.inference_mode()
def main():
    args = _parse_args()
    # Exhaustive sweeps can compile millions of one-use artifacts.
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    device = torch.device("cuda", args.device)
    identity = _identity(device, args.n, args.k, args)
    if identity["gfx"] not in SUPPORTED_ARCHS:
        raise SystemExit(f"MXFP4 W4A16 is unsupported on {identity['gfx']}")

    history = {m: {} for m in args.m}
    columns = [*identity, "m", *_CONFIG_FIELDS, *_RESULT_FIELDS]
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
    csv_file = (
        args.csv.open("a+", encoding="utf-8", newline="", errors="surrogateescape")
        if args.csv
        else nullcontext(None)
    )
    with torch.cuda.device(device), csv_file as handle:
        writer = None
        if handle is not None:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise SystemExit(f"Another tuner is using {args.csv}") from error
            has_header, loaded = _read_csv(
                handle, columns, identity, history, repair_tail=True
            )
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            if not has_header:
                writer.writeheader()
                handle.flush()
            if loaded:
                print(f"Restored {loaded:,} trials from {args.csv}", flush=True)
            if args.continue_from and not args.csv.samefile(args.continue_from):
                with args.continue_from.open(
                    encoding="utf-8", newline="", errors="surrogateescape"
                ) as source:
                    try:
                        fcntl.flock(source, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    except BlockingIOError as error:
                        raise SystemExit(
                            f"Another tuner is writing {args.continue_from}"
                        ) from error
                    _, copied = _read_csv(
                        source, columns, identity, history, copy_to=writer
                    )
                handle.flush()
                print(
                    f"Imported {copied:,} trials from {args.continue_from}", flush=True
                )
        winners, failed = _sweep(args, device, identity, writer, handle, history)
        if args.output is not None and winners:
            _write_winners(args.output, winners)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
