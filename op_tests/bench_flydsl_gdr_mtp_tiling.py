"""A/B the MTP tiling rule against a tuned table, in one process.

The rule in ``_mtp_rung`` was fitted on gfx942. This is what says whether it
holds on another part: it runs both arms against the same tensors, alternating
which goes first so drift over the run does not settle on one of them, and
reports each against the upstream the contract is measured by, so parity can be
read per cell rather than inferred from an average.

    # rule against the table as it ships
    python op_tests/bench_flydsl_gdr_mtp_tiling.py --out /tmp/ab.json

    # rule against a table from before the rows were dropped
    python op_tests/bench_flydsl_gdr_mtp_tiling.py --table <pre-change-csv>

Without ``--table`` the shapes come from the rows the installed table has for
this part, which is empty once the rows are dropped; point ``--table`` at the
csv that still has them to get both the shape list and the arm to compare to.
"""

import argparse
import csv
import json
import math
import statistics
import sys
import time

import torch

import aiter
from aiter.ops.flydsl import linear_attention_kernels as LA
from op_tests.test_flydsl_gdr_mtp import test_gdr_mtp_perf

# The table's name for a contract, and the mode the perf row wants for it.
VARIANT_MODE = {
    "snapshot": "sglang_chain",
    "snapshot_tree": "sglang_tree",
    "chain": "vllm_chain",
}
UPSTREAM = {"sglang_chain": "sglang", "sglang_tree": "sglang", "vllm_chain": "vllm"}
DTYPE = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}


def _table(path):
    """The rows for this part, as ``(shapes, map)``."""
    LA._tuned_config("torch.bfloat16", "torch.bfloat16", 1, 1, 1, 1, 128, 128, "decode")
    if path is None:
        return LA.GDR_GLOBAL_CONFIG_MAP, dict(LA.GDR_GLOBAL_CONFIG_MAP)
    rows = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["arch"] != LA.GDR_GPU_ARCH or row["variant"] not in VARIANT_MODE:
                continue
            cfg = {
                "NUM_BLOCKS_PER_V_DIM": int(row["NUM_BLOCKS_PER_V_DIM"]),
                "NUM_WARPS": int(row["NUM_WARPS"]),
                "WARP_THREADS_K": int(row["WARP_THREADS_K"]),
            }
            if row.get("waves_per_eu"):
                cfg["WAVES_PER_EU"] = int(row["waves_per_eu"])
            rows[
                (
                    row["dtype"],
                    row["state_dtype"],
                    row["arch"],
                    row["variant"],
                    int(row["b"]),
                    int(row["sq"]),
                    int(row["num_k_heads"]),
                    int(row["num_v_heads"]),
                    int(row["head_k_dim"]),
                    int(row["head_v_dim"]),
                )
            ] = cfg
    return rows, rows


def _shapes(rows):
    out = []
    for key in rows:
        d_str, sd_str, _, variant, b, sq, _, hv, khd, vhd = key
        if variant not in VARIANT_MODE or khd != 128 or vhd != 128:
            continue
        out.append(
            {
                "mode": VARIANT_MODE[variant],
                "variant": variant,
                "b": b,
                "sq": sq,
                "hv": hv,
                "dtype": d_str,
                "state_dtype": sd_str,
            }
        )
    return sorted(out, key=lambda s: (s["mode"], s["b"], s["sq"], s["hv"]))


def _run(shape, repeats):
    """One arm: the config the installed table yields, and the row it times."""
    LA._mtp_kwargs.cache_clear()
    got = LA.get_mtp_default_kwargs(
        shape["dtype"],
        shape["state_dtype"],
        DTYPE[shape["state_dtype"]],
        shape["b"],
        shape["sq"],
        max(1, shape["hv"] // 2),
        shape["hv"],
        128,
        128,
        shape["variant"],
    )
    row = test_gdr_mtp_perf(
        batch=shape["b"],
        seqlen=shape["sq"],
        mode=shape["mode"],
        num_v_heads=shape["hv"],
        dtype=DTYPE[shape["dtype"]],
        state_dtype=DTYPE[shape["state_dtype"]],
        repeats=repeats,
    )
    cfg = (
        f"{got['NUM_BLOCKS_PER_V_DIM']},{got['NUM_WARPS']},{got['WARP_THREADS_K']}"
        f",{got.get('WAVES_PER_EU', 0)}"
    )
    return cfg, float(row["flydsl us"]), float(row[f"{UPSTREAM[shape['mode']]} us"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", help="csv to use as the table arm and shape list")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", help="write the per-cell numbers here")
    opt = ap.parse_args()

    rows, full = _table(opt.table)
    shapes = _shapes(rows)
    if not shapes:
        sys.exit(
            f"no MTP rows for {LA.GDR_GPU_ARCH} in "
            f"{opt.table or 'the installed table'}; pass --table"
        )
    print(f"{len(shapes)} shapes on {LA.GDR_GPU_ARCH}, {opt.repeats} repeats\n")
    print(
        f"{'cell':<30} {'rule':<11} {'table':<11} {'rule us':>9} {'table us':>9} "
        f"{'r/t':>6} {'upstream':>9} {'r/up':>6}"
    )

    out, t0 = [], time.time()
    for i, shape in enumerate(shapes):
        arms = {}
        for arm in ("rule", "table") if i % 2 == 0 else ("table", "rule"):
            LA.GDR_GLOBAL_CONFIG_MAP = {} if arm == "rule" else dict(full)
            arms[arm] = _run(shape, opt.repeats)
        (rcfg, rus, rup), (tcfg, tus, tup) = arms["rule"], arms["table"]
        up = min(rup, tup)
        key = f"{shape['mode']} b{shape['b']} sq{shape['sq']} hv{shape['hv']}"
        print(
            f"{key:<30} {rcfg:<11} {tcfg:<11} {rus:9.2f} {tus:9.2f} {rus / tus:6.3f} "
            f"{up:9.2f} {rus / up:6.3f}" + ("  <<<" if rus > min(tus, up) else "")
        )
        out.append(
            {
                "cell": key,
                "rule_config": rcfg,
                "table_config": tcfg,
                "rule_us": rus,
                "table_us": tus,
                "upstream_us": up,
            }
        )
    LA.GDR_GLOBAL_CONFIG_MAP = full

    def geo(xs):
        return math.exp(statistics.fmean(math.log(x) for x in xs))

    rt = [c["rule_us"] / c["table_us"] for c in out]
    ru = [c["rule_us"] / c["upstream_us"] for c in out]
    tu = [c["table_us"] / c["upstream_us"] for c in out]
    print(f"\n{len(out)} cells in {time.time() - t0:.0f}s")
    print(f"  rule / table     geomean {geo(rt):.4f}  worst {max(rt):.3f}")
    print(
        f"  rule / upstream  geomean {geo(ru):.4f}  worst {max(ru):.3f}  "
        f"slower than upstream: {sum(x > 1 for x in ru)}"
    )
    print(
        f"  table / upstream geomean {geo(tu):.4f}  worst {max(tu):.3f}  "
        f"slower than upstream: {sum(x > 1 for x in tu)}"
    )
    if opt.out:
        with open(opt.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        aiter.logger.info("wrote %s", opt.out)


if __name__ == "__main__":
    main()
