# SPDX-License-Identifier: MIT
"""Sweep configs of AITER's Triton FA varlen_fwd at a fixed ViT shape.

For each (BLOCK_M, BLOCK_N, num_warps, num_stages, waves_per_eu, PRE_LOAD_V)
config, fork bench_fwd_prefill_single_config.py with the config set via
FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON. A fresh process is needed per
config because the env is read at module import time by
aiter.ops.triton._triton_kernels.flash_attn_triton_amd.utils.

Default sweep is an axis sweep around the AITER hardcoded RDNA default
(BM=128, BN=32, NW=8, ns=1, WE=6, PRE_LOAD_V=False), useful for spot-
checking whether the hardcoded config is still optimal on a new arch.

Output: JSON list at <out_dir>/sweep_results.json + per-config json/log.
"""

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO = Path(__file__).parent
BENCH = REPO / "bench_fwd_prefill_single_config.py"

# AITER's hardcoded RDNA default
# (aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py:67-79).
BASE = {
    "BLOCK_M": 128, "BLOCK_N": 32, "PRE_LOAD_V": False,
    "num_stages": 1, "num_warps": 8, "waves_per_eu": 6,
}


def _gpu_lock_cmd(cmd: list[str]) -> list[str]:
    g = shutil.which("gpu-lock")
    return ([g] + cmd) if g else cmd


def _shape_args(args) -> list[str]:
    return [
        "--batch", str(args.batch),
        "--seq", str(args.seq),
        "--heads", str(args.heads),
        "--head-dim", str(args.head_dim),
        "--dtype", args.dtype,
    ]


def _shape_tag(args) -> str:
    return (
        f"b{args.batch}_s{args.seq}_h{args.heads}_d{args.head_dim}_{args.dtype}"
    )


def axis_grid() -> list[dict]:
    """Vary one knob at a time around BASE."""
    grid: list[dict] = [dict(BASE)]
    # Bare default (no waves_per_eu) for completeness
    g0 = dict(BASE); g0.pop("waves_per_eu", None); grid.append(g0)
    for bm in (16, 32, 64, 128):
        c = dict(BASE); c["BLOCK_M"] = bm; grid.append(c)
    for bn in (16, 32, 64, 128):
        c = dict(BASE); c["BLOCK_N"] = bn; grid.append(c)
    for nw in (2, 4, 8):
        c = dict(BASE); c["num_warps"] = nw; grid.append(c)
    for ns in (1, 2, 3):
        c = dict(BASE); c["num_stages"] = ns; grid.append(c)
    for we in (1, 2, 4, 6, 8):
        c = dict(BASE); c["waves_per_eu"] = we; grid.append(c)
    for pl in (False, True):
        c = dict(BASE); c["PRE_LOAD_V"] = pl; grid.append(c)
    seen, uniq = set(), []
    for c in grid:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key); uniq.append(c)
    return uniq


def cross_grid(seed_configs: list[dict]) -> list[dict]:
    """Small cross-product around given seed configs."""
    bms = sorted({c["BLOCK_M"] for c in seed_configs} | {64, 128})
    bns = sorted({c["BLOCK_N"] for c in seed_configs} | {32, 64})
    nws = sorted({c["num_warps"] for c in seed_configs} | {4, 8})
    nss = sorted({c["num_stages"] for c in seed_configs} | {1})
    wes = sorted({c.get("waves_per_eu") for c in seed_configs if c.get("waves_per_eu") is not None} | {2, 6})
    pls = sorted({c["PRE_LOAD_V"] for c in seed_configs} | {False})
    return [
        {"BLOCK_M": bm, "BLOCK_N": bn, "PRE_LOAD_V": pl,
         "num_stages": ns, "num_warps": nw, "waves_per_eu": we}
        for bm, bn, nw, ns, we, pl in itertools.product(bms, bns, nws, nss, wes, pls)
    ]


def run_one(cfg: dict, shape_args: list[str], out_dir: Path) -> dict:
    tag = (
        f"bm{cfg['BLOCK_M']}_bn{cfg['BLOCK_N']}_nw{cfg['num_warps']}"
        f"_ns{cfg['num_stages']}_we{cfg.get('waves_per_eu')}"
        f"_pl{int(cfg['PRE_LOAD_V'])}"
    )
    log = out_dir / f"{tag}.log"
    js = out_dir / f"{tag}.json"
    env = os.environ.copy()
    env["FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON"] = json.dumps(cfg)
    env["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"

    cmd = _gpu_lock_cmd([sys.executable, str(BENCH)] + shape_args)
    t0 = time.time()
    with open(log, "w") as f:
        p = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=f, env=env, timeout=600,
        )
    elapsed = time.time() - t0
    if p.returncode != 0:
        return {"tag": tag, "error": f"rc={p.returncode} log={log}", "elapsed": elapsed}
    lines = p.stdout.decode().strip().splitlines()
    if not lines:
        return {"tag": tag, "error": f"no stdout log={log}", "elapsed": elapsed}
    try:
        data = json.loads(lines[-1])
    except json.JSONDecodeError as e:
        return {"tag": tag, "error": f"bad json: {e} log={log}", "elapsed": elapsed}
    js.write_text(json.dumps(data, indent=2))
    return {"tag": tag, "config": cfg, **data, "elapsed": elapsed}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--phase", choices=("axis", "cross", "custom"), default="axis")
    p.add_argument("--seed-configs", default=None, help="for cross: JSON list")
    p.add_argument("--configs", default=None, help="for custom: JSON list")
    # Shape args (forwarded to bench)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=3200)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=72)
    p.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    p.add_argument("--out", default=None, help="output directory")
    args = p.parse_args()

    if args.phase == "axis":
        grid = axis_grid()
    elif args.phase == "cross":
        with open(args.seed_configs) as f:
            seeds = json.load(f)
        grid = cross_grid(seeds)
    else:
        with open(args.configs) as f:
            grid = json.load(f)

    out_dir = Path(args.out) if args.out else Path(
        os.environ.get(
            "AITER_FA_SWEEP_OUT",
            str(Path(tempfile.gettempdir()) / "aiter_triton_fa_sweep"),
        )
    ) / _shape_tag(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sweep_results.json"
    shape_args = _shape_args(args)

    print(f"Shape: {_shape_tag(args)}", file=sys.stderr)
    print(f"Out dir: {out_dir}", file=sys.stderr)
    print(f"Configs to sweep: {len(grid)}", file=sys.stderr)

    results: list[dict] = []
    for i, cfg in enumerate(grid):
        line = (
            f"[{i+1:>3}/{len(grid)}] "
            f"BM={cfg['BLOCK_M']:>3} BN={cfg['BLOCK_N']:>3} "
            f"NW={cfg['num_warps']} ns={cfg['num_stages']} "
            f"WE={cfg.get('waves_per_eu')} pl={int(cfg['PRE_LOAD_V'])}"
        )
        print(line, file=sys.stderr, flush=True)
        r = run_one(cfg, shape_args, out_dir)
        results.append(r)
        out_path.write_text(json.dumps(results, indent=2))
        if "error" in r:
            print(f"  ERROR: {r['error']}", file=sys.stderr)
        else:
            print(
                f"  median={r['median_ms']:.3f} ms  min={r['min_ms']:.3f}",
                file=sys.stderr,
            )

    ok = [r for r in results if "error" not in r]
    ok.sort(key=lambda r: r["median_ms"])
    print(f"\nTop {min(15, len(ok))}:", file=sys.stderr)
    for r in ok[:15]:
        c = r["config"]
        print(
            f"  BM={c['BLOCK_M']:>3} BN={c['BLOCK_N']:>3} NW={c['num_warps']} "
            f"ns={c['num_stages']} WE={c.get('waves_per_eu')} "
            f"pl={int(c['PRE_LOAD_V'])}  "
            f"median={r['median_ms']:.3f} ms  min={r['min_ms']:.3f}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
