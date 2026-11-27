# SPDX-License-Identifier: MIT

r"""Sweep RDNA FlyDSL MXFP4 W4A16 configs for a list of M values at fixed N and K.

    python -m op_tests.tuners.tune_mxfp4_w4a16 --m 1-5 --n 34816 --k 5120 \
        --csv quark-gate-up-trials.csv

The default search crosses every launch knob. Tile M/N and stage K use
16, 32, 64, and 128. Traversal covers every
distinct group width for the shape. Use --space to replace any axis (for
example, --space stage_k=16-256:16). This can be a long sweep. Use --max-trials
to run it in batches and reuse --csv to resume. Use --slow-cutoff to screen
slow configs after the first timing batch and --trial-timeout to bound each
trial's wall time. --continue-from imports a
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
import signal
import statistics
from contextlib import nullcontext
from dataclasses import asdict, fields, replace
from itertools import product
from multiprocessing import get_context
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
_TRIAL_STATUSES = (
    "ok",
    "compile_error",
    "invalid",
    "incorrect",
    "out_of_memory",
    "screened_slow",
    "timed_out",
)


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


def _read_csv(
    handle, columns, identity, history, args, *, repair_tail=False, copy_to=None
):
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
            if status == "screened_slow":
                screen = json.loads(row["detail"])
                if not (
                    args.slow_cutoff
                    and screen["space"] == args.screen_space
                    and screen["batch_us"] > args.slow_cutoff * screen["best_us"]
                ):
                    continue
            if status == "timed_out" and not (
                0 < args.trial_timeout <= json.loads(row["detail"])["timeout_seconds"]
            ):
                continue
            if config in history[m] and (
                copy_to is not None
                or (latency is None and history[m][config] is not None)
            ):
                continue
            if copy_to is not None:
                copy_to.writerow(row)
            history[m][config] = latency
            loaded += 1
        except (ValueError, TypeError, KeyError, csv.Error) as error:
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


def _measure(a, prepared, out, expected, config, args, best_us):
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
    for batch in range(args.batches):
        start.record()
        for _ in range(args.replays):
            graph.replay()
        end.record()
        end.synchronize()
        latency = start.elapsed_time(end) * 1000 / (args.replays * args.graph_gemms)
        if not math.isfinite(latency) or latency <= 0:
            raise RuntimeError(f"Invalid measured latency: {latency}")
        if (
            batch == 0
            and args.batches > 1
            and args.slow_cutoff
            and best_us is not None
            and latency > args.slow_cutoff * best_us
        ):
            detail = json.dumps(
                {"batch_us": latency, "best_us": best_us, "space": args.screen_space}
            )
            return "screened_slow", None, detail
        samples.append(latency)
    return "ok", statistics.median(samples), ""


def _trial(shape, problem, prepared, config, args, best_us):
    a, out, expected = problem
    try:
        return _measure(a, prepared, out, expected, config, args, best_us)
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


def _trial_worker(connection, device, args):
    # Include any external compiler children in the watchdog's process group.
    os.setsid()
    with connection, torch.cuda.device(device), torch.inference_mode():
        weights = _make_weights(args.n, args.k, device)
        problem = None
        while True:
            try:
                shape, config, best_us = connection.recv()
            except EOFError:
                return
            if config is None:
                problem = _make_problem(shape, device, *weights[:2])
                torch.cuda.synchronize(device)
                connection.send(None)
            else:
                connection.send(
                    _trial(shape, problem, weights[2], config, args, best_us)
                )


class _TimedTrials:
    """Keep HIP compilation in a worker that can be replaced after a timeout."""

    def __init__(self, device, args):
        self.device = device
        self.args = args
        self.context = get_context("spawn")
        self.process = None
        self.connection = None
        self.shape = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc, _traceback):
        self._close(force=exc_type is not None)

    def _close(self, *, force=False):
        if self.process is None:
            return
        self.connection.close()
        if not force:
            self.process.join(timeout=2)
        if self.process.is_alive():
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                self.process.kill()
        self.process.join()
        self.process.close()
        self.process = self.connection = self.shape = None

    def _receive(self):
        try:
            return self.connection.recv()
        except EOFError as error:
            raise RuntimeError(
                "The tuning worker exited before returning a result"
            ) from error

    def run(self, shape, config, best_us):
        if self.process is None:
            parent, child = self.context.Pipe()
            process = self.context.Process(
                target=_trial_worker, args=(child, self.device, self.args)
            )
            try:
                process.start()
            except BaseException:
                parent.close()
                raise
            finally:
                child.close()
            self.process, self.connection = process, parent
        if self.shape != shape:
            self.connection.send((shape, None, None))
            if not self.connection.poll(max(120, self.args.trial_timeout)):
                raise RuntimeError(
                    f"The tuning worker timed out preparing inputs for {shape}"
                )
            self._receive()
            self.shape = shape

        self.connection.send((shape, config, best_us))
        if self.connection.poll(self.args.trial_timeout):
            return self._receive()
        self._close(force=True)
        return (
            "timed_out",
            None,
            json.dumps({"timeout_seconds": self.args.trial_timeout}),
        )


def _restore_summary(history, geometries, schedule):
    if not history:
        return 0, None
    widths_by_geometry = {
        tuple(getattr(config, name) for name in _GEOMETRY): widths
        for config, widths in geometries
    }
    count = 0
    best = None
    for config, latency in history.items():
        widths = widths_by_geometry.get(
            tuple(getattr(config, name) for name in _GEOMETRY), ()
        )
        if config.traversal_group_m not in widths or any(
            getattr(config, name) not in values for name, values in schedule.items()
        ):
            continue
        count += 1
        if latency is not None and (best is None or latency < best[0]):
            best = (latency, config)
    return count, best


def _sweep(args, device, identity, writer, handle, history, worker):
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
        restored, best = _restore_summary(history[m], geometries, schedule)
        problem = None
        tested = skipped = rejected = 0
        screened = timed_out = 0
        last_error = ""
        complete = True
        with tqdm(
            total=total,
            initial=restored,
            desc=f"M={m} N={args.n} K={args.k}",
            unit="cfg",
            dynamic_ncols=True,
            postfix=f"resumed={restored:,}, slow=0, timeout=0",
        ) as progress:
            for config in _configs(geometries, schedule):
                if config in history[m]:
                    latency = history[m][config]
                    skipped += 1
                else:
                    if args.max_trials is not None and new_trials >= args.max_trials:
                        complete = False
                        break
                    best_us = best[0] if best is not None else None
                    if worker is not None:
                        status, latency, detail = worker.run(shape, config, best_us)
                    else:
                        if weights is None:
                            weights = _make_weights(args.n, args.k, device)
                        if problem is None:
                            problem = _make_problem(shape, device, *weights[:2])
                        status, latency, detail = _trial(
                            shape, problem, weights[2], config, args, best_us
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
                    if status in ("screened_slow", "timed_out"):
                        screened += status == "screened_slow"
                        timed_out += status == "timed_out"
                        progress.set_postfix_str(
                            f"resumed={restored:,}, slow={screened:,}, timeout={timed_out:,}",
                            refresh=False,
                        )
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
    parser.add_argument(
        "--slow-cutoff",
        type=float,
        default=0,
        metavar="RATIO",
        help="Skip remaining batches if the first exceeds RATIO times the best verified config (0 disables)",
    )
    parser.add_argument(
        "--trial-timeout",
        type=float,
        default=0,
        metavar="SECONDS",
        help="Wall-time limit per trial, including compilation (0 disables)",
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
    if not math.isfinite(args.slow_cutoff) or (
        args.slow_cutoff != 0 and args.slow_cutoff <= 1
    ):
        parser.error("--slow-cutoff must be 0 or a finite ratio greater than 1")
    if not math.isfinite(args.trial_timeout) or args.trial_timeout < 0:
        parser.error("--trial-timeout must be finite and non-negative")
    if args.slow_cutoff:
        search_space = _GEOMETRY | _SCHEDULE | args.axes
        args.screen_space = hashlib.sha256(
            json.dumps(search_space, sort_keys=True).encode()
        ).hexdigest()
    else:
        args.screen_space = None
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
    timed_trials = (
        _TimedTrials(device, args) if args.trial_timeout else nullcontext(None)
    )
    with torch.cuda.device(device), csv_file as handle, timed_trials as worker:
        writer = None
        if handle is not None:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise SystemExit(f"Another tuner is using {args.csv}") from error
            has_header, loaded = _read_csv(
                handle, columns, identity, history, args, repair_tail=True
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
                        source, columns, identity, history, args, copy_to=writer
                    )
                handle.flush()
                print(
                    f"Imported {copied:,} trials from {args.continue_from}", flush=True
                )
        winners, failed = _sweep(
            args, device, identity, writer, handle, history, worker
        )
        if args.output is not None and winners:
            _write_winners(args.output, winners)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
