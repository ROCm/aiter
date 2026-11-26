# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Accuracy sweep of the gfx942 FlyDSL 8-wave FP8 blockscale kernel against the
kernel each tuned table currently names for the same shape.

Run from the repository root on a gfx942 GPU:
  python scripts/check_blockscale_gemm_gfx942.py --collect-only
  python scripts/check_blockscale_gemm_gfx942.py --device 2 --shard 0 --num-shards 4
  python scripts/check_blockscale_gemm_gfx942.py --summary results_*.csv

--collect-only (or a missing --shapes file) collects every gfx942 row of the
plain and B-preshuffle FP32-scale tuned tables under aiter/configs, including
model_configs (MX-scale, group32, A-preshuffle and untuned tables excluded),
deduplicated by M,N,K,preshuffle. Each shape keeps one tuned row as its
baseline, preferring cu_num 80 (MI308), then 304, then the others.

Method (scripts/check_blockscale_gemm.py of the gfx950 kernel, commit 99cfe72):
A and B are torch.randn cast to FP8 (E4M3FNUZ on gfx942), scales are rand + 0.5.
The reference uses the SAME FP8 values dequantized with the FP32 scales, rounded
to BF16 and multiplied by F.linear. calc_diff = 1 - 2*sum(x*y)/sum(x*x+y*y) with
FP64 sums; smaller is better and <= --threshold passes. The same reference
scores the baseline (the tuned row's kernel run through the public op with the
tuned lookup injected) and the FlyDSL kernel, so their diffs are comparable.
That reference's BF16 operand rounding dominates both diffs (about 5e-6);
--reference fp32 keeps the dequantized operands and the GEMM in FP32 instead,
so each diff is the kernel's own error (BF16 output rounding is about 1.3e-6).

To make 25k+ shapes affordable, the shapes of one (N, K, layout) share inputs:
A, its scales and the reference are built once for the largest M of the group
and the first M rows are used for every M (row i of the result only depends on
row i of A). Each group gets a seed derived from --seed, N, K and the layout.
FlyDSL is called directly (no CK/ASM fallback) into a NaN-filled output with
guard elements on both sides; an unwritten element, a store outside the output
or a nonfinite value fails. --repeats N reruns FlyDSL on the same inputs and
requires bitwise identical outputs. Statistics are reduced by one fused Triton
pass in FP64.

A persistent spawned worker owns the GPU. On exceptions, crashes or per-case
timeouts it is replaced before the next shape. Results are flushed after each
case; exit status is 1 if any FlyDSL row is not PASS, 2 for setup errors.
"""

import argparse
import csv
import math
import multiprocessing as mp
import sys
import time
import traceback
from collections import Counter, defaultdict
from multiprocessing.connection import Connection
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHAPES = Path(__file__).with_name("blockscale_gemm_shapes_gfx942.csv")
FAMILIES = (
    (False, "a8w8_blockscale_tuned_gemm"),
    (True, "a8w8_blockscale_bpreshuffle_tuned_gemm"),
)
# Legacy tables without a gfx column, as aiter's config merge backfills them.
CU_NUM_GFX = {80: "gfx942", 304: "gfx942", 256: "gfx950"}
BASE_FIELDS = ("cu_num", "libtype", "kernelId", "splitK", "kernelName", "source")
SHAPE_FIELDS = ("M", "N", "K", "preshuffle") + BASE_FIELDS + ("tuned_rows",)
RESULT_FIELDS = (
    "M",
    "N",
    "K",
    "preshuffle",
    "status",
    "calc_diff",
    "base_calc_diff",
    "diff_ratio",
    "fly_vs_base_calc_diff",
    "max_abs",
    "base_max_abs",
    "rmse",
    "base_rmse",
    "finite",
    "base_finite",
    "guard_ok",
    "deterministic",
    "repeats_completed",
    "base_status",
    "base_libtype",
    "base_kernelId",
    "base_splitK",
    "base_kernelName",
    "base_cu_num",
    "base_source",
    "tuned_rows",
    "seed",
    "reference",
    "threshold",
    "device",
    "error",
)
GUARD = 256
SENTINEL = 42.0


class Shape(NamedTuple):
    m: int
    n: int
    k: int
    preshuffle: bool
    base: tuple  # BASE_FIELDS values; empty strings when no baseline
    tuned_rows: int


def _clean(value) -> str:
    return "" if value is None else str(value).strip()


def collect_shapes(config_dir: Path, output: Path) -> None:
    rows: dict[tuple, list] = defaultdict(list)
    sources = []
    for ps, family in FAMILIES:
        paths = sorted(
            p for p in config_dir.rglob(f"*{family}*.csv") if "untuned" not in p.name
        )
        if not paths:
            raise ValueError(f"{config_dir}: no tables matching *{family}*.csv")
        for path in paths:
            sources.append(path)
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                reader.fieldnames = [c.strip() for c in reader.fieldnames or []]
                for line, row in enumerate(reader, 2):
                    try:
                        cu_num = int(row["cu_num"])
                        gfx = _clean(row.get("gfx")) or CU_NUM_GFX.get(cu_num, "")
                        if gfx != "gfx942":
                            continue
                        m, n, k = (int(row[c]) for c in ("M", "N", "K"))
                        base = (
                            cu_num,
                            _clean(row.get("libtype")),
                            _clean(row.get("kernelId")),
                            _clean(row.get("splitK")),
                            _clean(row.get("kernelName")),
                            str(path.relative_to(config_dir)),
                        )
                        rows[(m, n, k, ps)].append(base)
                    except (KeyError, ValueError, TypeError) as exc:
                        raise ValueError(f"{path}:{line}: {exc}") from exc
    if output.resolve() in {p.resolve() for p in sources}:
        raise ValueError("Shape output must not overwrite a tuned source table")
    # MI308 rows first: they are the ones this kernel can be tuned in for.
    order = {80: 0, 304: 1}
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SHAPE_FIELDS)
        for (m, n, k, ps), bases in sorted(
            rows.items(), key=lambda kv: (kv[0][3], kv[0][1], kv[0][2], kv[0][0])
        ):
            base = min(bases, key=lambda b: (order.get(b[0], 2), b[5]))
            writer.writerow((m, n, k, int(ps), *base, len(bases)))
    counts = Counter(key[3] for key in rows)
    print(
        f"Collected {len(rows)} gfx942 shapes from {len(sources)} tables: "
        f"plain={counts[False]}, preshuffle={counts[True]} -> {output}",
        flush=True,
    )


def read_shapes(path: Path) -> list[Shape]:
    shapes = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not set(SHAPE_FIELDS[:4]).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing required columns {SHAPE_FIELDS[:4]}")
        for line, row in enumerate(reader, 2):
            try:
                m, n, k = (int(row[c]) for c in ("M", "N", "K"))
                if min(m, n, k) <= 0:
                    raise ValueError("M, N and K must be positive")
                flag = row["preshuffle"].strip().lower()
                if flag not in ("0", "1", "false", "true"):
                    raise ValueError("preshuffle must be 0/1/false/true")
                base = tuple(_clean(row.get(c)) for c in BASE_FIELDS)
                tuned_rows = int(row.get("tuned_rows") or 0)
                shapes.append(Shape(m, n, k, flag in ("1", "true"), base, tuned_rows))
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"{path}:{line}: {exc}") from exc
    if not shapes:
        raise ValueError(f"{path}: no shapes")
    return shapes


def group_seed(base_seed: int, n: int, k: int, ps: bool) -> int:
    return (base_seed + n * 1000003 + k * 1009 + int(ps)) % (2**63)


def shard_shapes(shapes: list[Shape], shard: int, num_shards: int) -> list[Shape]:
    """Whole (N, K, layout) groups, balanced by output elements + GEMM work."""
    groups: dict[tuple, list[Shape]] = defaultdict(list)
    for s in shapes:
        groups[(s.n, s.k, s.preshuffle)].append(s)
    cost = {
        key: sum(s.m * s.n * (1 + s.k / 2048) for s in grp)
        for key, grp in groups.items()
    }
    loads = [0.0] * num_shards
    owner = {}
    for key in sorted(groups, key=lambda g: -cost[g]):
        target = loads.index(min(loads))
        owner[key] = target
        loads[target] += cost[key]
    mine = [key for key in groups if owner[key] == shard]
    mine.sort(key=lambda g: (g[2], g[0], g[1]))
    return [s for key in mine for s in sorted(groups[key], key=lambda s: s.m)]


_STATS_KERNEL = None


def _stats_kernel():
    """One FP64 pass: FlyDSL vs ref, baseline vs ref and FlyDSL vs baseline."""
    global _STATS_KERNEL
    if _STATS_KERNEL is not None:
        return _STATS_KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def stats(
        x_ptr,
        y_ptr,
        r_ptr,
        part_ptr,
        n,
        HAS_Y: tl.constexpr,
        BLOCK: tl.constexpr,
        ITERS: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        start = pid * (BLOCK * ITERS)
        zero = tl.zeros((BLOCK,), dtype=tl.float64)
        x_xy, x_den, x_dd, x_max, x_bad = zero, zero, zero, zero, zero
        y_xy, y_den, y_dd, y_max, y_bad = zero, zero, zero, zero, zero
        c_xy, c_den, c_dd, c_max, r_bad = zero, zero, zero, zero, zero
        for i in range(ITERS):
            offs = start + i * BLOCK + tl.arange(0, BLOCK)
            msk = offs < n
            r = tl.load(r_ptr + offs, mask=msk, other=0.0).to(tl.float32)
            r = r.to(tl.float64)
            # v - v is 0 for finite v and NaN for NaN/inf.
            r_ok = (r - r) == 0.0
            r_bad += tl.where(r_ok, 0.0, 1.0)
            x = tl.load(x_ptr + offs, mask=msk, other=0.0).to(tl.float32)
            x = x.to(tl.float64)
            x_ok = (x - x) == 0.0
            x_bad += tl.where(x_ok, 0.0, 1.0)
            ok = x_ok & r_ok
            xs = tl.where(ok, x, 0.0)
            rs = tl.where(ok, r, 0.0)
            d = xs - rs
            x_xy += xs * rs
            x_den += xs * xs + rs * rs
            x_dd += d * d
            x_max = tl.maximum(x_max, tl.abs(d))
            if HAS_Y:
                y = tl.load(y_ptr + offs, mask=msk, other=0.0).to(tl.float32)
                y = y.to(tl.float64)
                y_ok = (y - y) == 0.0
                y_bad += tl.where(y_ok, 0.0, 1.0)
                ok = y_ok & r_ok
                ys = tl.where(ok, y, 0.0)
                rs = tl.where(ok, r, 0.0)
                d = ys - rs
                y_xy += ys * rs
                y_den += ys * ys + rs * rs
                y_dd += d * d
                y_max = tl.maximum(y_max, tl.abs(d))
                ok = x_ok & y_ok
                xs = tl.where(ok, x, 0.0)
                ys = tl.where(ok, y, 0.0)
                d = xs - ys
                c_xy += xs * ys
                c_den += xs * xs + ys * ys
                c_dd += d * d
                c_max = tl.maximum(c_max, tl.abs(d))
        out = part_ptr + pid * 16
        tl.store(out + 0, tl.sum(x_xy, 0))
        tl.store(out + 1, tl.sum(x_den, 0))
        tl.store(out + 2, tl.sum(x_dd, 0))
        tl.store(out + 3, tl.max(x_max, 0))
        tl.store(out + 4, tl.sum(x_bad, 0))
        tl.store(out + 5, tl.sum(y_xy, 0))
        tl.store(out + 6, tl.sum(y_den, 0))
        tl.store(out + 7, tl.sum(y_dd, 0))
        tl.store(out + 8, tl.max(y_max, 0))
        tl.store(out + 9, tl.sum(y_bad, 0))
        tl.store(out + 10, tl.sum(c_xy, 0))
        tl.store(out + 11, tl.sum(c_den, 0))
        tl.store(out + 12, tl.sum(c_dd, 0))
        tl.store(out + 13, tl.max(c_max, 0))
        tl.store(out + 14, tl.sum(r_bad, 0))
        tl.store(out + 15, 0.0)

    _STATS_KERNEL = stats
    return stats


def calc_diff(xy: float, den: float, bad: float) -> float:
    if bad:
        return float("inf")
    return 1.0 - 2.0 * xy / den if den else 0.0


def compare(actual, baseline, reference) -> dict:
    """calc_diff/max_abs/rmse of actual (and baseline) against reference."""
    import torch

    block, iters = 1024, 64
    count = reference.numel()
    programs = max(1, -(-count // (block * iters)))
    part = torch.empty((programs, 16), dtype=torch.float64, device=reference.device)
    has_y = baseline is not None
    _stats_kernel()[(programs,)](
        actual,
        baseline if has_y else actual,
        reference,
        part,
        count,
        HAS_Y=has_y,
        BLOCK=block,
        ITERS=iters,
    )
    sums = part.sum(0).tolist()
    maxes = part.amax(0).tolist()
    r_bad = sums[14]
    result = {
        "calc_diff": calc_diff(sums[0], sums[1], sums[4] + r_bad),
        "max_abs": maxes[3] if not sums[4] + r_bad else float("inf"),
        "rmse": math.sqrt(sums[2] / count) if not sums[4] + r_bad else float("inf"),
        "finite": not sums[4],
        "ref_finite": not r_bad,
    }
    if has_y:
        result.update(
            base_calc_diff=calc_diff(sums[5], sums[6], sums[9] + r_bad),
            base_max_abs=maxes[8] if not sums[9] + r_bad else float("inf"),
            base_rmse=(
                math.sqrt(sums[7] / count) if not sums[9] + r_bad else float("inf")
            ),
            base_finite=not sums[9],
            fly_vs_base_calc_diff=calc_diff(sums[10], sums[11], sums[4] + sums[9]),
        )
    return result


class GroupData:
    """Inputs and reference shared by every M of one (N, K, layout, seed)."""

    def __init__(self, n, k, ps, m_max, seed, device, ref_rows, ref_dtype):
        import torch
        import torch.nn.functional as F

        from aiter import dtypes
        from aiter.ops.shuffle import shuffle_weight

        self.key = (n, k, ps, m_max, seed, ref_dtype)
        kb = k // 128
        gen = torch.Generator(device=device).manual_seed(seed)
        fp8 = dtypes.fp8
        self.a = torch.empty((m_max, k), dtype=fp8, device=device)
        for row in range(0, m_max, 8192):
            end = min(row + 8192, m_max)
            self.a[row:end] = torch.randn(
                (end - row, k), device=device, generator=gen
            ).to(fp8)
        b = torch.randn((n, k), device=device, generator=gen).to(fp8)
        self.sa = torch.rand((m_max, kb), device=device, generator=gen) + 0.5
        self.sb = torch.rand(((n + 127) // 128, kb), device=device, generator=gen) + 0.5
        b_ref = torch.empty((n, k), device=device, dtype=ref_dtype)
        for row in range(0, n, 128):
            end = min(row + 128, n)
            b_ref[row:end] = (
                (
                    b[row:end].float().reshape(end - row, kb, 128)
                    * self.sb[row // 128, :, None]
                )
                .reshape(end - row, k)
                .to(ref_dtype)
            )
        self.ref = torch.empty((m_max, n), device=device, dtype=ref_dtype)
        for row in range(0, m_max, ref_rows):
            end = min(row + ref_rows, m_max)
            a_ref = (
                (
                    self.a[row:end].float().reshape(end - row, kb, 128)
                    * self.sa[row:end, :, None]
                )
                .reshape(end - row, k)
                .to(ref_dtype)
            )
            self.ref[row:end] = F.linear(a_ref, b_ref)
        del b_ref
        self.weight = shuffle_weight(b, layout=(16, 16)) if ps else b
        # FlyDSL output with guard elements on both sides; base output for the
        # preshuffle op's out= (the plain op allocates its own).
        self.fly_buf = torch.full(
            (m_max * n + 2 * GUARD,), SENTINEL, device=device, dtype=torch.bfloat16
        )
        self.base_buf = (
            torch.empty((m_max * n,), device=device, dtype=torch.bfloat16)
            if ps
            else None
        )


def run_baseline(shape: Shape, x, weight, x_scale, w_scale, out):
    """The tuned row's kernel, as the public op serves it."""
    from unittest import mock

    import aiter
    from aiter import dtypes
    from aiter.ops import gemm_op_a8w8

    _cu_num, libtype, kernel_id, split_k, kernel_name, _src = shape.base
    config = {
        "libtype": libtype,
        "kernelId": int(float(kernel_id)) if kernel_id else 0,
        "splitK": int(float(split_k)) if split_k else 0,
        "kernelName": kernel_name,
    }
    with mock.patch.object(gemm_op_a8w8, "get_CKGEMM_config", lambda *a, **kw: config):
        if shape.preshuffle:
            return aiter.gemm_a8w8_blockscale_bpreshuffle(
                x, weight, x_scale, w_scale, dtypes.bf16, out=out
            )
        return aiter.gemm_a8w8_blockscale(x, weight, x_scale, w_scale, dtypes.bf16)


def run_case(shape: Shape, m_max: int, args, state: dict) -> dict:
    import torch

    from aiter.ops.flydsl.gemm_a8w8_blockscale import run_gemm_a8w8_blockscale
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_blockscale_common import (
        kernel_fits_shape,
        kernelInstance,
    )

    m, n, k, ps = shape.m, shape.n, shape.k, shape.preshuffle
    ker_obj = kernelInstance(ps, "gfx942")
    if not kernel_fits_shape(ker_obj, m, n, k, "gfx942"):
        return {"status": "UNSUPPORTED", "error": "Outside the gfx942 shape contract"}
    device = torch.device("cuda", args.device)
    seed = group_seed(args.seed, n, k, ps)
    ref_dtype = torch.float32 if args.reference == "fp32" else torch.bfloat16
    data = state.get("data")
    if data is None or data.key != (n, k, ps, m_max, seed, ref_dtype):
        state["data"] = data = None
        torch.cuda.empty_cache()
        state["data"] = data = GroupData(
            n, k, ps, m_max, seed, device, args.reference_rows, ref_dtype
        )
    kb = k // 128
    x = data.a[:m]
    x_scale = data.sa[:m]
    # bpreshuffle contract: x_scale bytes are [K/128, M] behind an (M, K/128) view.
    if ps:
        x_scale = x_scale.T.contiguous().view(m, kb)
    reference = data.ref[:m]
    buf = data.fly_buf
    out = buf[GUARD : GUARD + m * n].view(m, n)
    buf[:GUARD].fill_(SENTINEL)
    buf[GUARD + m * n : 2 * GUARD + m * n].fill_(SENTINEL)

    def launch():
        out.fill_(float("nan"))
        run_gemm_a8w8_blockscale(
            x, data.weight, x_scale, data.sb, out, ker_obj.name, preshuffle_b=ps
        )

    launch()
    torch.cuda.synchronize(device)
    base_out = None
    base_status = "NONE"
    base_error = ""
    if shape.base[1]:
        try:
            base_out = run_baseline(
                shape,
                x,
                data.weight,
                x_scale,
                data.sb,
                data.base_buf[: m * n].view(m, n) if ps else None,
            )
            torch.cuda.synchronize(device)
            base_status = "OK"
        except Exception as exc:  # record and keep FlyDSL's result
            base_status = "ERROR"
            base_error = f"baseline: {type(exc).__name__}: {exc}"[:500]
            if "HIP" in str(exc) or "hipError" in str(exc):
                raise
    stats = compare(out, base_out, reference)
    guard_ok = bool(
        (buf[:GUARD] == SENTINEL).all()
        and (buf[GUARD + m * n : 2 * GUARD + m * n] == SENTINEL).all()
    )
    deterministic = True
    first = out.clone() if args.repeats > 1 else None
    for _ in range(1, args.repeats):
        launch()
        torch.cuda.synchronize(device)
        deterministic &= bool(
            torch.equal(out.view(torch.int16), first.view(torch.int16))
        )
        stats["calc_diff"] = max(
            stats["calc_diff"], compare(out, None, reference)["calc_diff"]
        )
    if not stats["ref_finite"]:
        raise RuntimeError("nonfinite reference")
    result = {
        "status": "PASS",
        "calc_diff": stats["calc_diff"],
        "max_abs": stats["max_abs"],
        "rmse": stats["rmse"],
        "finite": stats["finite"],
        "guard_ok": guard_ok,
        "deterministic": deterministic,
        "repeats_completed": args.repeats,
        "seed": seed,
        "reference": args.reference,
        "base_status": base_status,
        "error": base_error,
    }
    if base_out is not None:
        base_diff = stats["base_calc_diff"]
        result.update(
            base_calc_diff=base_diff,
            base_max_abs=stats["base_max_abs"],
            base_rmse=stats["base_rmse"],
            base_finite=stats["base_finite"],
            fly_vs_base_calc_diff=stats["fly_vs_base_calc_diff"],
            diff_ratio=(
                stats["calc_diff"] / base_diff
                if base_diff and math.isfinite(base_diff)
                else ""
            ),
        )
    if not (
        stats["finite"]
        and guard_ok
        and deterministic
        and stats["calc_diff"] <= args.threshold
    ):
        result["status"] = "FAIL"
    return result


def worker(connection: Connection, args) -> None:
    sys.path.insert(0, str(ROOT))
    try:
        import torch

        from aiter.jit.utils.chip_info import get_gfx_runtime

        torch.cuda.set_device(args.device)
        gfx = get_gfx_runtime()
        if gfx != "gfx942":
            raise RuntimeError(f"This sweep needs gfx942, got {gfx}")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_tf32 = False
        connection.send({"status": "READY", "device": torch.cuda.get_device_name()})
    except Exception:  # noqa: BLE001 - transport setup traceback to the parent
        connection.send({"status": "ERROR", "error": traceback.format_exc()})
        return
    state: dict = {}
    with torch.inference_mode():
        while True:
            job = connection.recv()
            if job is None:
                return
            shape, m_max = job
            try:
                connection.send(run_case(shape, m_max, args, state))
            except Exception:  # noqa: BLE001 - isolate arbitrary JIT/HIP failures
                # A failed HIP call may poison the context. Report, then replace it.
                connection.send({"status": "ERROR", "error": traceback.format_exc()})
                return


def receive(connection: Connection, timeout: float) -> dict:
    if not connection.poll(timeout):
        return {"status": "TIMEOUT", "error": f"No response within {timeout:g}s"}
    try:
        return connection.recv()
    except (EOFError, ConnectionError) as exc:
        return {"status": "CRASH", "error": f"GPU worker exited: {exc!r}"}


def stop_worker(process, connection: Connection) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
    connection.close()
    process.close()


def read_done(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        return {
            (int(r["M"]), int(r["N"]), int(r["K"]), r["preshuffle"] == "1")
            for r in csv.DictReader(f)
        }


def sweep(shapes: list[Shape], args) -> int:
    context = mp.get_context("spawn")
    process = connection = None
    counts: Counter[str] = Counter()
    m_max = defaultdict(int)
    for s in shapes:
        key = (s.n, s.k, s.preshuffle)
        m_max[key] = max(m_max[key], s.m)
    done = read_done(args.results) if args.resume else set()
    todo = [s for s in shapes if (s.m, s.n, s.k, s.preshuffle) not in done]
    if done:
        print(f"Resuming: {len(shapes) - len(todo)} shapes already in {args.results}")
    started = time.monotonic()
    mode = "a" if args.resume and args.results.exists() else "w"
    with args.results.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if mode == "w":
            writer.writeheader()
        f.flush()
        try:
            for index, shape in enumerate(todo, 1):
                if process is None:
                    connection, child = context.Pipe()
                    process = context.Process(target=worker, args=(child, args))
                    process.start()
                    child.close()
                    ready = receive(connection, args.timeout)
                    if ready["status"] != "READY":
                        raise RuntimeError(f"GPU worker setup failed: {ready}")
                    print(f"GPU {args.device}: {ready['device']}", flush=True)
                key = (shape.n, shape.k, shape.preshuffle)
                try:
                    connection.send((shape, m_max[key]))
                    result = receive(connection, args.timeout)
                except (BrokenPipeError, ConnectionError) as exc:
                    result = {"status": "CRASH", "error": repr(exc)}
                base = dict(zip(BASE_FIELDS, shape.base))
                row = {
                    "M": shape.m,
                    "N": shape.n,
                    "K": shape.k,
                    "preshuffle": int(shape.preshuffle),
                    "threshold": args.threshold,
                    "device": args.device,
                    "tuned_rows": shape.tuned_rows,
                    **{f"base_{c}": base[c] for c in BASE_FIELDS},
                }
                row.update(result)
                writer.writerow(row)
                f.flush()
                counts[result["status"]] += 1
                if index % args.log_every == 0 or result["status"] != "PASS":
                    eta = (time.monotonic() - started) / index * (len(todo) - index)
                    print(
                        f"[{index}/{len(todo)}] {shape.m}x{shape.n}x{shape.k} "
                        f"ps={int(shape.preshuffle)} {result['status']} "
                        f"diff={result.get('calc_diff', '-')} "
                        f"base={result.get('base_calc_diff', '-')} "
                        f"pass={counts['PASS']} issues={index - counts['PASS']} "
                        f"ETA={eta / 60:.1f}min",
                        flush=True,
                    )
                if result.get("error") and result["status"] != "PASS":
                    print(result["error"], file=sys.stderr, flush=True)
                if result["status"] in ("ERROR", "CRASH", "TIMEOUT"):
                    stop_worker(process, connection)
                    process = connection = None
        finally:
            if process is not None:
                try:
                    connection.send(None)
                except (BrokenPipeError, ConnectionError):
                    pass
                stop_worker(process, connection)
    print(f"Summary: {dict(counts)}; results: {args.results}", flush=True)
    return int(counts["PASS"] != len(todo))


def summarize(paths: list[Path]) -> int:
    import pandas as pd

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df.drop_duplicates(["M", "N", "K", "preshuffle"], keep="last")
    print(f"{len(df)} shapes from {len(paths)} result files")
    print("\nFlyDSL status:", df["status"].value_counts().to_dict())
    print("Baseline status:", df["base_status"].value_counts().to_dict())
    ok = df[df["base_status"] == "OK"].copy()
    for col in ("calc_diff", "base_calc_diff", "diff_ratio", "fly_vs_base_calc_diff"):
        ok[col] = pd.to_numeric(ok[col], errors="coerce")
    ok["base_split"] = pd.to_numeric(ok["base_splitK"], errors="coerce").fillna(0) > 0
    rows = []
    for (ps, lib, split), grp in ok.groupby(
        ["preshuffle", "base_libtype", "base_split"]
    ):
        rows.append(
            {
                "layout": "preshuffle" if ps else "plain",
                "baseline": lib + (" splitK>0" if split else ""),
                "shapes": len(grp),
                "flydsl diff max": grp["calc_diff"].max(),
                "flydsl diff mean": grp["calc_diff"].mean(),
                "base diff max": grp["base_calc_diff"].max(),
                "base diff mean": grp["base_calc_diff"].mean(),
                "ratio median": grp["diff_ratio"].median(),
                "ratio max": grp["diff_ratio"].max(),
                "flydsl<=base": int((grp["calc_diff"] <= grp["base_calc_diff"]).sum()),
                f"base>{df['threshold'].iloc[0]:g}": int(
                    (grp["base_calc_diff"] > grp["threshold"]).sum()
                ),
                "bitwise equal": int((grp["fly_vs_base_calc_diff"] == 0).sum()),
                "fly-vs-base max": grp["fly_vs_base_calc_diff"].max(),
            }
        )
    if rows:
        print("\n" + pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3g"))
    bad = df[df["status"] != "PASS"]
    if len(bad):
        print("\nNon-PASS FlyDSL rows:")
        print(
            bad[["M", "N", "K", "preshuffle", "status", "calc_diff", "error"]]
            .head(50)
            .to_string(index=False)
        )
    worse = ok[ok["diff_ratio"] > 1.05]
    print(f"\nFlyDSL diff > 1.05x baseline diff: {len(worse)} shapes")
    if len(worse):
        cols = [
            "M",
            "N",
            "K",
            "preshuffle",
            "calc_diff",
            "base_calc_diff",
            "diff_ratio",
            "base_libtype",
            "base_splitK",
        ]
        print(
            worse.sort_values("diff_ratio", ascending=False)[cols]
            .head(30)
            .to_string(index=False)
        )
    return int(len(bad) > 0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--shapes", type=Path, default=DEFAULT_SHAPES)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "aiter/configs")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument(
        "--summary", type=Path, nargs="+", help="Summarize result CSVs and exit"
    )
    parser.add_argument("--results", type=Path)
    parser.add_argument("--resume", action="store_true", help="Skip rows in --results")
    parser.add_argument(
        "--device", type=int, default=0, help="Visible CUDA/HIP device index"
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument(
        "--reference",
        choices=("bf16", "fp32"),
        default="bf16",
        help="bf16: BF16-rounded operands (gfx950 sweep's method); fp32: exact",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base of the group seeds")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--timeout", type=float, default=900, help="Seconds per case/startup"
    )
    parser.add_argument("--reference-rows", type=int, default=8192)
    parser.add_argument("--limit", type=int, help="Maximum rows of this shard")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if args.summary:
        return summarize(args.summary)
    if not math.isfinite(args.threshold) or args.threshold <= 0:
        parser.error("--threshold must be finite and positive")
    if min(args.repeats, args.reference_rows, args.num_shards, args.log_every) < 1:
        parser.error("--repeats/--reference-rows/--num-shards must be positive")
    if not 0 <= args.shard < args.num_shards or args.device < 0:
        parser.error("--shard must be in [0, --num-shards); --device nonnegative")
    try:
        if args.collect_only or not args.shapes.exists():
            collect_shapes(args.config_dir, args.shapes)
        if args.collect_only:
            return 0
        if args.results is None:
            args.results = Path(f"blockscale_results_gfx942_shard{args.shard}.csv")
        if args.results.resolve() == args.shapes.resolve():
            raise ValueError("--results must not overwrite --shapes")
        shapes = shard_shapes(read_shapes(args.shapes), args.shard, args.num_shards)
        shapes = shapes[: args.limit] if args.limit else shapes
        if not shapes:
            raise ValueError("No shapes selected")
        print(f"Testing {len(shapes)} shapes; results -> {args.results}", flush=True)
        return sweep(shapes, args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
