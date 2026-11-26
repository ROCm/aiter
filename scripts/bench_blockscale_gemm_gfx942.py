# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Time the gfx942 FlyDSL 8-wave FP8 blockscale kernel against the kernel each
tuned row of this GPU (gfx942, --cu-num, 80 = MI308X) names, and switch the rows
FlyDSL wins to it.

Run from the repository root:
  python scripts/bench_blockscale_gemm_gfx942.py --device 2 --shard 0 --num-shards 4
  python scripts/bench_blockscale_gemm_gfx942.py --summary bench_gfx942_*.csv \
      [--accuracy blockscale_results_gfx942_*.csv]
  python scripts/bench_blockscale_gemm_gfx942.py --apply bench_gfx942_*.csv

Rows come from the plain and B-preshuffle FP32-scale tuned tables under
aiter/configs (canonical and model_configs). Each row is measured as the tuner
measures a candidate: its inputs (rand/10 FP8, rand scales, seed 0), run_perftest
with --warmup/--iters and rotating input copies (GPU kernel time), and errRatio
from checkAllclose(rtol=atol=1e-2) against the FP32 reference. The row's own
kernel and the FlyDSL kernel (kernelId 2 plain B, 3 preshuffled B) both run
through the public op with the tuned lookup injected, alternately for --rounds
rounds; each keeps its fastest round. calc_diff (1 - 2*sum(x*y)/sum(x*x+y*y))
of both against the BF16-rounded reference is recorded too.

A row switches to FlyDSL when FlyDSL is at least --min-gain faster and its
errRatio is within --err-ratio. --summary and --apply combine every result file
they are given: a row timed in several runs keeps each kernel's fastest time
and the decision is re-derived. --apply rewrites only the switched rows of
their source tables (libtype, kernelId, splitK, us, kernelName, tflops, bw,
errRatio); every other line is kept byte for byte. --model-configs-only skips
the canonical tables.
"""

import argparse
import csv
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = (
    (False, "a8w8_blockscale_tuned_gemm"),
    (True, "a8w8_blockscale_bpreshuffle_tuned_gemm"),
)
CU_NUM_GFX = {80: "gfx942", 304: "gfx942", 256: "gfx950"}
RESULT_FIELDS = (
    "source",
    "M",
    "N",
    "K",
    "preshuffle",
    "cu_num",
    "base_libtype",
    "base_kernelId",
    "base_splitK",
    "base_kernelName",
    "csv_us",
    "base_us",
    "flydsl_us",
    "speedup",
    "replace",
    "base_errRatio",
    "flydsl_errRatio",
    "base_calc_diff",
    "flydsl_calc_diff",
    "flydsl_vs_base_calc_diff",
    "flydsl_kernelId",
    "flydsl_kernelName",
    "device",
    "error",
)


def _clean(value) -> str:
    return "" if value is None else str(value).strip()


def collect_rows(config_dir: Path, cu_num: int, model_configs_only: bool) -> list:
    rows = []
    for ps, family in FAMILIES:
        for path in sorted(config_dir.rglob(f"*{family}*.csv")):
            if "untuned" in path.name:
                continue
            if model_configs_only and path.parent.name != "model_configs":
                continue
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                reader.fieldnames = [c.strip() for c in reader.fieldnames or []]
                for row in reader:
                    row_cu = int(row["cu_num"])
                    gfx = _clean(row.get("gfx")) or CU_NUM_GFX.get(row_cu, "")
                    if gfx != "gfx942" or row_cu != cu_num:
                        continue
                    rows.append(
                        {
                            "source": str(path.relative_to(ROOT)),
                            "M": int(row["M"]),
                            "N": int(row["N"]),
                            "K": int(row["K"]),
                            "preshuffle": int(ps),
                            "cu_num": row_cu,
                            "base_libtype": _clean(row["libtype"]),
                            "base_kernelId": _clean(row["kernelId"]),
                            "base_splitK": _clean(row["splitK"]),
                            "base_kernelName": _clean(row["kernelName"]),
                            "csv_us": _clean(row["us"]),
                        }
                    )
    return rows


def shard_rows(rows: list, shard: int, num_shards: int) -> list:
    # Round-robin over an M*N*K order spreads the large shapes evenly.
    ordered = sorted(rows, key=lambda r: (r["M"] * r["N"] * r["K"], r["source"]))
    return [r for i, r in enumerate(ordered) if i % num_shards == shard]


def generate_data(m, n, k, seed, device):
    """gemm_a8w8_blockscale_tune.py's inputs."""
    import torch

    from aiter import dtypes
    from aiter.ops.shuffle import shuffle_weight

    torch.manual_seed(seed)
    scale_n, scale_k = (n + 127) // 128, (k + 127) // 128
    x = (torch.rand((m, k), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
    x_scale = torch.rand([m, scale_k], dtype=dtypes.fp32, device=device)
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device=device)
    return {
        "x": x,
        "weight": weight,
        "x_scale": x_scale,
        "w_scale": w_scale,
        "weight_shuffle": shuffle_weight(weight, layout=(16, 16)),
        "x_scale_t": x_scale.transpose(0, 1).contiguous().view(*x_scale.shape),
    }


def run_torch(x, weight, x_scale, w_scale):
    """gemm_a8w8_blockscale_tune.py's FP32 reference, BF16 output."""
    import torch
    import torch.nn.functional as F

    m, k = x.shape
    n = weight.shape[0]
    kb = k // 128
    a = (x.float().view(m, kb, 128) * x_scale.unsqueeze(-1)).view(m, k)
    b = (
        weight.float().view(n, kb, 128)
        * w_scale.repeat_interleave(128, dim=0)[:n, :, None]
    ).view(n, k)
    return F.linear(a, b).to(torch.bfloat16)


def bench_row(row: dict, args) -> dict:
    from unittest import mock

    import torch
    from check_blockscale_gemm_gfx942 import compare

    import aiter
    from aiter import dtypes
    from aiter.ops import gemm_op_a8w8
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_blockscale_common import (
        kernel_fits_shape,
        kernels_list,
    )
    from aiter.test_common import checkAllclose, run_perftest

    m, n, k, ps = row["M"], row["N"], row["K"], bool(row["preshuffle"])
    fly_id = 3 if ps else 2
    fly = kernels_list[fly_id]
    result = {"flydsl_kernelId": fly_id, "flydsl_kernelName": fly.name}
    if not kernel_fits_shape(fly, m, n, k, "gfx942"):
        return {**result, "error": "Outside the gfx942 FlyDSL shape contract"}
    device = torch.device("cuda", args.device)
    data = generate_data(m, n, k, 0, device)
    ref = run_torch(data["x"], data["weight"], data["x_scale"], data["w_scale"])
    weight = data["weight_shuffle"] if ps else data["weight"]
    x_scale = data["x_scale_t"] if ps else data["x_scale"]
    configs = {
        "base": {
            "libtype": row["base_libtype"],
            "kernelId": int(float(row["base_kernelId"] or 0)),
            "splitK": int(float(row["base_splitK"] or 0)),
            "kernelName": row["base_kernelName"],
        },
        "flydsl": {
            "libtype": "flydsl",
            "kernelId": fly_id,
            "splitK": 0,
            "kernelName": fly.name,
        },
    }

    def op(x, w, xs, ws, out):
        if ps:
            return aiter.gemm_a8w8_blockscale_bpreshuffle(
                x, w, xs, ws, dtypes.bf16, out=out
            )
        return aiter.gemm_a8w8_blockscale(x, w, xs, ws, dtypes.bf16)

    outs, times = {}, defaultdict(list)
    for _ in range(args.rounds):
        for tag, config in configs.items():
            out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
            with mock.patch.object(
                gemm_op_a8w8,
                "get_CKGEMM_config",
                lambda *a, _config=config, **kw: _config,
            ):
                res, us = run_perftest(
                    op,
                    data["x"],
                    weight,
                    x_scale,
                    data["w_scale"],
                    out,
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                )
            torch.cuda.synchronize(device)
            times[tag].append(us)
            outs[tag] = res
    for tag in configs:
        result[f"{'flydsl' if tag == 'flydsl' else 'base'}_errRatio"] = round(
            checkAllclose(ref, outs[tag], rtol=1e-2, atol=1e-2, printLog=False), 4
        )
    stats = compare(outs["flydsl"], outs["base"], ref)
    base_us, fly_us = min(times["base"]), min(times["flydsl"])
    result.update(
        base_us=round(base_us, 4),
        flydsl_us=round(fly_us, 4),
        speedup=round(base_us / fly_us, 4) if fly_us > 0 else "",
        base_calc_diff=stats["base_calc_diff"],
        flydsl_calc_diff=stats["calc_diff"],
        flydsl_vs_base_calc_diff=stats["fly_vs_base_calc_diff"],
    )
    result["replace"] = int(
        0 < fly_us <= base_us * (1 - args.min_gain)
        and result["flydsl_errRatio"] <= args.err_ratio
    )
    return result


def bench(rows: list, args) -> int:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

    torch.cuda.set_device(args.device)
    if get_gfx_runtime() != "gfx942" or get_cu_num() != args.cu_num:
        raise RuntimeError(
            f"Needs gfx942 with cu_num {args.cu_num}, got "
            f"{get_gfx_runtime()} cu_num {get_cu_num()}"
        )
    done = set()
    if args.resume and args.results.exists():
        with args.results.open(newline="") as f:
            done = {
                (r["source"], int(r["M"]), int(r["N"]), int(r["K"]))
                for r in csv.DictReader(f)
            }
    todo = [r for r in rows if (r["source"], r["M"], r["N"], r["K"]) not in done]
    mode = "a" if done else "w"
    started = time.monotonic()
    failures = 0
    with args.results.open(mode, newline="") as f, torch.inference_mode():
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if mode == "w":
            writer.writeheader()
        for index, row in enumerate(todo, 1):
            out = {**row, "device": args.device}
            try:
                out.update(bench_row(row, args))
            except Exception:  # noqa: BLE001 - record and keep going
                out["error"] = traceback.format_exc()[-800:]
                failures += 1
            torch.cuda.empty_cache()
            writer.writerow(out)
            f.flush()
            eta = (time.monotonic() - started) / index * (len(todo) - index)
            print(
                f"[{index}/{len(todo)}] {row['M']}x{row['N']}x{row['K']} "
                f"ps={row['preshuffle']} {row['base_libtype']} "
                f"base={out.get('base_us', '-')} flydsl={out.get('flydsl_us', '-')} "
                f"x{out.get('speedup', '-')} replace={out.get('replace', '-')} "
                f"ETA={eta / 60:.1f}min",
                flush=True,
            )
            if out.get("error"):
                print(out["error"], file=sys.stderr, flush=True)
    return int(failures > 0)


def read_results(paths: list, args) -> "pd.DataFrame":
    """Rows of every run; a row timed in several runs keeps each kernel's
    fastest time, and the switch decision is re-derived from those."""
    import pandas as pd

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    key = ["source", "M", "N", "K"]
    ok = df[df["error"].isna()]
    agg = {c: "first" for c in df.columns if c not in key}
    agg.update(
        base_us="min", flydsl_us="min", base_errRatio="max", flydsl_errRatio="max"
    )
    ok = ok.groupby(key, as_index=False, sort=False).agg(agg)
    ok["speedup"] = (ok["base_us"] / ok["flydsl_us"]).round(4)
    ok["replace"] = (
        (ok["flydsl_us"] <= ok["base_us"] * (1 - args.min_gain))
        & (ok["flydsl_errRatio"] <= args.err_ratio)
    ).astype(int)
    failed = df[df["error"].notna()].drop_duplicates(key, keep="last")
    failed = failed[~failed.set_index(key).index.isin(ok.set_index(key).index)]
    return pd.concat([ok, failed], ignore_index=True)


def summarize(paths: list, accuracy: list | None, args) -> int:
    import numpy as np
    import pandas as pd

    df = read_results(paths, args)
    ok = df[df["error"].isna()].copy()
    ok["replace"] = ok["replace"].astype(int) == 1
    ok["new_us"] = np.where(ok["replace"], ok["flydsl_us"], ok["base_us"])
    print(f"{len(df)} rows, {len(df) - len(ok)} errors")
    rows = []
    for key, grp in list(ok.groupby("source")) + [("ALL", ok)]:
        rep = grp[grp["replace"]]
        rows.append(
            {
                "table": Path(key).name,
                "rows": len(grp),
                "switched": len(rep),
                "old sum us": grp["base_us"].sum(),
                "new sum us": grp["new_us"].sum(),
                "total speedup": grp["base_us"].sum() / grp["new_us"].sum(),
                "switched geomean": (
                    float(np.exp(np.log(rep["speedup"]).mean())) if len(rep) else ""
                ),
                "switched max": rep["speedup"].max() if len(rep) else "",
                "flydsl faster (any)": int((grp["speedup"] > 1).sum()),
            }
        )
    print("\n" + pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4g"))
    rows = []
    for lib, grp in ok.groupby("base_libtype"):
        rep = grp[grp["replace"]]
        rows.append(
            {
                "old libtype": lib,
                "rows": len(grp),
                "switched": len(rep),
                "speedup median": grp["speedup"].median(),
                "switched geomean": (
                    float(np.exp(np.log(rep["speedup"]).mean())) if len(rep) else ""
                ),
                "old errRatio max": grp["base_errRatio"].max(),
                "flydsl errRatio max": grp["flydsl_errRatio"].max(),
                "old calc_diff mean": grp["base_calc_diff"].mean(),
                "flydsl calc_diff mean": grp["flydsl_calc_diff"].mean(),
            }
        )
    print("\n" + pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4g"))
    by_m = ok.assign(
        bucket=pd.cut(
            ok["M"],
            [0, 64, 256, 1024, 4096, 16384, 1 << 30],
            labels=["<=64", "65-256", "257-1024", "1025-4096", "4097-16384", ">16384"],
        )
    )
    rows = []
    for bucket, grp in by_m.groupby("bucket", observed=True):
        rows.append(
            {
                "M": bucket,
                "rows": len(grp),
                "switched": int(grp["replace"].sum()),
                "speedup median": grp["speedup"].median(),
                "speedup max": grp["speedup"].max(),
            }
        )
    print("\n" + pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4g"))
    if accuracy:
        acc = pd.concat([pd.read_csv(p) for p in accuracy], ignore_index=True)
        acc = acc[
            ["M", "N", "K", "preshuffle", "status", "calc_diff", "base_calc_diff"]
        ]
        acc = acc.rename(
            columns={"calc_diff": "acc_flydsl_diff", "base_calc_diff": "acc_base_diff"}
        )
        merged = ok.merge(acc, on=["M", "N", "K", "preshuffle"], how="left")
        rows = []
        for (lib, rep), grp in merged.groupby(["base_libtype", "replace"]):
            rows.append(
                {
                    "old libtype": lib,
                    "switched": rep,
                    "rows": len(grp),
                    "accuracy PASS": int((grp["status"] == "PASS").sum()),
                    "old calc_diff mean": grp["acc_base_diff"].mean(),
                    "old calc_diff max": grp["acc_base_diff"].max(),
                    "flydsl calc_diff mean": grp["acc_flydsl_diff"].mean(),
                    "flydsl calc_diff max": grp["acc_flydsl_diff"].max(),
                }
            )
        print("\ncalc_diff of the same rows from the accuracy sweep (randn data):")
        print(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3g"))
    return 0


def _fmt(value: float, digits: int) -> str:
    text = f"{round(float(value), digits):.{digits}f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def apply(paths: list, args) -> int:
    df = read_results(paths, args)
    df = df[df["error"].isna() & (df["replace"].astype(float) == 1)]
    if args.model_configs_only:
        df = df[df["source"].str.contains("/model_configs/")]
    changed = 0
    for source, grp in df.groupby("source"):
        path = ROOT / source
        lines = path.read_text().splitlines(keepends=True)
        header = next(csv.reader([lines[0]]))
        cols = [c.strip() for c in header]
        want = {(int(r.M), int(r.N), int(r.K)): r for r in grp.itertuples()}
        hits = defaultdict(int)
        for i, line in enumerate(lines[1:], 1):
            if not line.strip():
                continue
            values = next(csv.reader([line]))
            row = dict(zip(cols, values))
            gfx = row.get("gfx", "").strip() or CU_NUM_GFX.get(int(row["cu_num"]), "")
            key = (int(row["M"]), int(row["N"]), int(row["K"]))
            if gfx != "gfx942" or int(row["cu_num"]) != args.cu_num or key not in want:
                continue
            r = want[key]
            m, n, k = key
            us = float(r.flydsl_us)
            row.update(
                libtype="flydsl",
                kernelId=str(int(r.flydsl_kernelId)),
                splitK="0",
                us=_fmt(us, 4),
                kernelName=r.flydsl_kernelName,
                tflops=_fmt(2 * m * n * k / (us * 1e6), 2),
                bw=_fmt((m * k + n * k + m * n * 2) / (us * 1e-6) / 1e9, 2),
                errRatio=_fmt(r.flydsl_errRatio, 4),
            )
            newline = "\r\n" if line.endswith("\r\n") else "\n"
            lines[i] = ",".join(row[c] for c in cols) + newline
            hits[key] += 1
        missing = set(want) - set(hits)
        dupes = [key for key, count in hits.items() if count > 1]
        if missing or dupes:
            raise ValueError(f"{source}: missing {sorted(missing)}, repeated {dupes}")
        path.write_text("".join(lines))
        changed += len(hits)
        print(f"{source}: switched {len(hits)} rows to FlyDSL")
    print(f"Switched {changed} rows in total")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config-dir", type=Path, default=ROOT / "aiter/configs")
    parser.add_argument("--cu-num", type=int, default=80)
    parser.add_argument("--model-configs-only", action="store_true")
    parser.add_argument("--summary", type=Path, nargs="+")
    parser.add_argument("--accuracy", type=Path, nargs="+")
    parser.add_argument("--apply", type=Path, nargs="+")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--rows", type=Path, help="Only rows whose source,M,N,K are in this CSV"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=101)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--min-gain", type=float, default=0.01, help="Minimum fractional time saving"
    )
    parser.add_argument("--err-ratio", type=float, default=0.05)
    args = parser.parse_args()
    if args.summary:
        return summarize(args.summary, args.accuracy, args)
    if args.apply:
        return apply(args.apply, args)
    if not 0 <= args.shard < args.num_shards or min(args.rounds, args.iters) < 1:
        parser.error("bad --shard/--num-shards/--rounds/--iters")
    rows = collect_rows(args.config_dir, args.cu_num, args.model_configs_only)
    if args.rows:
        with args.rows.open(newline="") as f:
            keep = {
                (r["source"], int(r["M"]), int(r["N"]), int(r["K"]))
                for r in csv.DictReader(f)
            }
        rows = [r for r in rows if (r["source"], r["M"], r["N"], r["K"]) in keep]
    rows = shard_rows(rows, args.shard, args.num_shards)[: args.limit]
    if args.results is None:
        args.results = Path(f"bench_gfx942_shard{args.shard}.csv")
    print(f"Timing {len(rows)} rows; results -> {args.results}", flush=True)
    return bench(rows, args)


if __name__ == "__main__":
    raise SystemExit(main())
