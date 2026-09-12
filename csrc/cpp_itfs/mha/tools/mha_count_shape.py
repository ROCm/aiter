# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
mha_count_shape.py -- aggregate MHA forward call shapes dumped by
AITER_DUMP_MHA_FWD_INFO.

Inspired by csrc/gemm_a16w16/countGemmShape.py. Split into two sub-commands:

  Step 1: group
      python mha_count_shape.py group -i mha_fwd_1.txt -d mha_logs/
    Behavior:
      - Parse the log and group records by
        (mode, dtype, hdim_q, hdim_v, mask_type).
      - For each group, emit **exactly one** CSV:
        mha_group_<id>_<sig>.csv
          * Rows are deduplicated by the full
            (seqlens_q, seqlens_k) tuple.
          * Each row = one unique shape combination + its occurrence count.
          * Columns: mode, dtype, hdim_q, hdim_v, mask_type, batch,
                     max_seqlen_q, total_q, total_k, seqlens_q, seqlens_k,
                     count
          * Row count = number of unique shape combinations in this group.
      - A summary CSV mha_groups_summary.csv is also written
        (group_id / signature / combo count / group CSV filename).
      - Terminal prints per-group unique seqlens and Top-K distributions,
        helping the user pick tuning ranges.

  Step 2: generate_tune_range
      python mha_count_shape.py generate_tune_range \
          -i mha_logs/out/mha_group_0_group_bf16_hq72_hv72_mask0.csv \
          --range 256:1024:64 --range 1024:4224:32 \
          --singletons 480,1600,2116,3128,3772,4056,4096,4104,4144,4176
    Behavior:
      - Read one group CSV produced by Step 1
        (mha_group_<gid>_<sig>.csv). The summary file is not required.
      - Extract (mode, dtype, hdim_q, hdim_v, mask_type) from the CSV
        first row.
      - --range S:E:STEP: may be given multiple times; each occurrence
        appends one closed-interval arithmetic segment.
      - --singletons a,b,c: extra discrete M values.
      - -o/--output: output CSV path. Defaults to the same directory as
        the input, with the filename prefix
        mha_group_*.csv -> mha_untune_*.csv.

Note: pure standard library, no pandas/numpy dependency.
"""

import argparse
import csv
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path


# --------------------------------------------------------------------------- #
# Log format (strictly aligned with the output emitted by mha_fwd_dump.h):
#   [MHA_FWD] mode=group dtype=bf16 hdim_q=72 hdim_v=72 nhead_q=16 nhead_k=16
#            batch=9 max_seqlen_q=4176 mask_type=0 bias_type=0 has_lse=0
#            has_dropout=0 total_q=29596 total_k=29596
#            seqlens_q=[988,1736,2500,3772,4144,4032,4144,4176,4104]
#            seqlens_k=[988,1736,2500,3772,4144,4032,4144,4176,4104]
# In batch mode, total_q/total_k/seqlens_* are absent and replaced by the
# scalars seqlen_q/seqlen_k.
# --------------------------------------------------------------------------- #
KV_RE = re.compile(r"(\w+)=([^\s\[]+|\[[^\]]*\])")
INT_LIST = re.compile(r"-?\d+")

GROUP_COLS = ("mode", "dtype", "hdim_q", "hdim_v", "mask_type")
SUMMARY_NAME = "mha_groups_summary.csv"


# --------------------------------------------------------------------------- #
# Common helpers
# --------------------------------------------------------------------------- #
def _parse_value(v: str):
    if v.startswith("["):
        return [int(x) for x in INT_LIST.findall(v)]
    try:
        return int(v)
    except ValueError:
        return v


def parse_log(input_path: Path):
    """Return list[dict], one per MHA forward call."""
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("[MHA_FWD]"):
                continue
            body = line[len("[MHA_FWD]") :].strip()
            rec = {k: _parse_value(v) for k, v in KV_RE.findall(body)}
            records.append(rec)
    return records


def group_signature(gkey):
    """Turn ('group','bf16',72,72,0) into 'group_bf16_hq72_hv72_mask0'."""
    mode, dtype, hq, hv, mt = gkey
    return f"{mode}_{dtype}_hq{hq}_hv{hv}_mask{mt}"


# --------------------------------------------------------------------------- #
# Step 1: group
# --------------------------------------------------------------------------- #
def cmd_group(args):
    input_path = Path(args.input_log).absolute()
    out_dir = Path(args.out_dir).absolute()
    if not input_path.is_file():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = parse_log(input_path)
    if not records:
        print(f"[ERROR] No [MHA_FWD] lines parsed from: {input_path}")
        sys.exit(1)
    print(f"[STAT] Parsed {len(records)} MHA forward calls from {input_path}")

    # ---------- Group-wise accumulation ---------- #
    #   groups[gkey] = {
    #       'num_calls', 'total_q_sum', 'total_k_sum',
    #       'batch_cnt', 'maxseq_cnt', 'seqs_q_cnt', 'seqs_q_tok',
    #       'seqs_k_cnt', 'seqs_k_tok',
    #       'shape_cnt': OrderedDict((seqlens_q_tuple, seqlens_k_tuple) -> {
    #             'count', 'batch', 'max_seqlen_q', 'total_q', 'total_k'})
    #   }
    groups = OrderedDict()
    for r in records:
        gkey = tuple(r.get(c) for c in GROUP_COLS)
        g = groups.setdefault(
            gkey,
            {
                "num_calls": 0,
                "total_q_sum": 0,
                "total_k_sum": 0,
                "batch_cnt": Counter(),
                "maxseq_cnt": Counter(),
                "seqs_q_cnt": Counter(),
                "seqs_q_tok": Counter(),
                "seqs_k_cnt": Counter(),
                "seqs_k_tok": Counter(),
                "shape_cnt": OrderedDict(),
            },
        )
        batch = r.get("batch", 1)
        tq = r.get("total_q")
        if tq is None:
            tq = batch * (r.get("seqlen_q") or 0)
        tk = r.get("total_k")
        if tk is None:
            tk = batch * (r.get("seqlen_k") or 0)

        g["num_calls"] += 1
        g["total_q_sum"] += tq
        g["total_k_sum"] += tk
        g["batch_cnt"][batch] += 1
        mq = r.get("max_seqlen_q", r.get("seqlen_q"))
        if mq is not None:
            g["maxseq_cnt"][mq] += 1

        seqs_q = r.get("seqlens_q")
        if seqs_q is None:
            seqs_q = (
                [r.get("seqlen_q")] * batch if r.get("seqlen_q") is not None else []
            )
        seqs_k = r.get("seqlens_k")
        if seqs_k is None:
            seqs_k = (
                [r.get("seqlen_k")] * batch if r.get("seqlen_k") is not None else []
            )
        for s in seqs_q:
            if s is None:
                continue
            g["seqs_q_cnt"][s] += 1
            g["seqs_q_tok"][s] += s
        for s in seqs_k:
            if s is None:
                continue
            g["seqs_k_cnt"][s] += 1
            g["seqs_k_tok"][s] += s

        # ---- Dedup key: full (seqlens_q, seqlens_k) tuple ---- #
        shape_key = (tuple(seqs_q), tuple(seqs_k))
        entry = g["shape_cnt"].get(shape_key)
        if entry is None:
            g["shape_cnt"][shape_key] = {
                "count": 1,
                "batch": batch,
                "max_seqlen_q": mq,
                "total_q": tq,
                "total_k": tk,
            }
        else:
            entry["count"] += 1

    # Sort groups by total_q_sum in descending order.
    sorted_groups = sorted(groups.items(), key=lambda kv: -kv[1]["total_q_sum"])
    grand_total = sum(g["total_q_sum"] for _, g in sorted_groups) or 1

    # ---------- Print overview ---------- #
    print(
        "\n================ GROUP SUMMARY "
        "(sorted by total_q_tokens desc) ================"
    )
    hdr = (
        ["gid"]
        + list(GROUP_COLS)
        + [
            "calls",
            "uniq_shape",
            "total_q_tokens",
            "%",
            "total_k_tokens",
            "uniq_seqQ",
            "uniq_maxSQ",
            "uniq_batch",
        ]
    )
    print("  " + "  ".join(f"{h:<12}" for h in hdr))
    for gid, (gkey, g) in enumerate(sorted_groups):
        ratio = g["total_q_sum"] / grand_total * 100
        cells = (
            [str(gid)]
            + [str(v) for v in gkey]
            + [
                str(g["num_calls"]),
                str(len(g["shape_cnt"])),
                str(g["total_q_sum"]),
                f"{ratio:.1f}%",
                str(g["total_k_sum"]),
                str(len(g["seqs_q_cnt"])),
                str(len(g["maxseq_cnt"])),
                str(len(g["batch_cnt"])),
            ]
        )
        print("  " + "  ".join(f"{c:<12}" for c in cells))

    # ---------- Per-group details + single CSV output ---------- #
    K = args.topk
    summary_rows = []
    for gid, (gkey, g) in enumerate(sorted_groups):
        sig = group_signature(gkey)
        header = "  ".join(f"{k}={v}" for k, v in zip(GROUP_COLS, gkey))
        print(f"\n---------------- GROUP[{gid}]: {header} ----------------")
        print(f"  num_calls={g['num_calls']}, uniq_shape_combos={len(g['shape_cnt'])}")

        uniq_max = sorted(g["maxseq_cnt"].keys())
        print(f"[INPUT max_seqlen_q] {len(uniq_max)} unique values: {uniq_max}")
        top_ms = sorted(g["maxseq_cnt"].items(), key=lambda x: -x[1])[:K]
        print(f"[max_seqlen_q by count] top{K}: {top_ms}")

        uniq_batch = sorted(g["batch_cnt"].keys())
        print(f"[INPUT batch]        {len(uniq_batch)} unique values: {uniq_batch}")

        uniq_sq = sorted(g["seqs_q_cnt"].keys())
        print(
            f"[INPUT seqlens_q]    {len(uniq_sq)} unique values"
            + ("" if len(uniq_sq) > 200 else f": {uniq_sq}")
        )
        top_qc = sorted(g["seqs_q_cnt"].items(), key=lambda x: -x[1])[:K]
        print(f"[seqlens_q by count] top{K}: {top_qc}")
        top_qt = sorted(g["seqs_q_tok"].items(), key=lambda x: -x[1])[:K]
        print(f"[seqlens_q by tokens] top{K}: {top_qt}")

        uniq_sk = sorted(g["seqs_k_cnt"].keys())
        print(
            f"[INPUT seqlens_k]    {len(uniq_sk)} unique values"
            + ("" if len(uniq_sk) > 200 else f": {uniq_sk}")
        )
        top_kc = sorted(g["seqs_k_cnt"].items(), key=lambda x: -x[1])[:K]
        print(f"[seqlens_k by count] top{K}: {top_kc}")
        top_kt = sorted(g["seqs_k_tok"].items(), key=lambda x: -x[1])[:K]
        print(f"[seqlens_k by tokens] top{K}: {top_kt}")

        # ---- Write: unique shape combinations for this group,
        #      sorted by count in descending order.               ---- #
        rec_path = out_dir / f"mha_group_{gid}_{sig}.csv"
        rec_cols = list(GROUP_COLS) + [
            "batch",
            "max_seqlen_q",
            "total_q",
            "total_k",
            "seqlens_q",
            "seqlens_k",
            "count",
        ]
        shape_items = sorted(
            g["shape_cnt"].items(),
            key=lambda kv: (-kv[1]["count"], -kv[1]["total_q"], -kv[1]["max_seqlen_q"]),
        )
        with rec_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rec_cols)
            w.writeheader()
            for (sq, sk), meta in shape_items:
                row = OrderedDict(zip(GROUP_COLS, gkey))
                row.update(
                    {
                        "batch": meta["batch"],
                        "max_seqlen_q": meta["max_seqlen_q"],
                        "total_q": meta["total_q"],
                        "total_k": meta["total_k"],
                        "seqlens_q": ",".join(str(x) for x in sq),
                        "seqlens_k": ",".join(str(x) for x in sk),
                        "count": meta["count"],
                    }
                )
                w.writerow(row)
        print(f"[WRITE] {rec_path}  ({len(shape_items)} unique combos)")

        summary_rows.append(
            OrderedDict(
                [
                    ("group_id", gid),
                    *((c, v) for c, v in zip(GROUP_COLS, gkey)),
                    ("signature", sig),
                    ("num_calls", g["num_calls"]),
                    ("uniq_shape_combos", len(g["shape_cnt"])),
                    ("total_q_tokens", g["total_q_sum"]),
                    ("total_k_tokens", g["total_k_sum"]),
                    ("uniq_seqlens_q", len(g["seqs_q_cnt"])),
                    ("uniq_max_seqlen_q", len(g["maxseq_cnt"])),
                    ("uniq_batch", len(g["batch_cnt"])),
                    ("group_csv", str(rec_path.name)),
                ]
            )
        )

    # ---------- Write summary ---------- #
    summary_path = out_dir / SUMMARY_NAME
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\n[WRITE] {summary_path}  ({len(summary_rows)} groups)")
    print("\nNext step example (run one command per group):")
    for srow in summary_rows:
        gcsv = out_dir / srow["group_csv"]
        print(
            f"  python {Path(__file__).name} generate_tune_range "
            f"-i {gcsv} "
            f"--range 512:2048:32 --range 2048:4224:64 "
            f"--singletons 4096,4104,4144,4176"
        )


# --------------------------------------------------------------------------- #
# Step 2: generate_tune_range
# --------------------------------------------------------------------------- #
def _parse_range(spec: str):
    """'S:E:STEP' -> range(S, E+1, STEP); the interval is closed."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--range expects 'S:E:STEP' format, got: {spec}"
        )
    try:
        s, e, st = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        raise argparse.ArgumentTypeError(f"--range values must be integers: {spec}")
    if st <= 0 or e < s:
        raise argparse.ArgumentTypeError(f"--range is invalid: {spec}")
    return list(range(s, e + 1, st))


def _parse_singletons(spec: str):
    if not spec:
        return []
    return [int(x) for x in spec.replace(" ", "").split(",") if x]


def cmd_generate_tune_range(args):
    in_csv = Path(args.input_csv).absolute()
    if not in_csv.is_file():
        print(f"[ERROR] Group CSV not found: {in_csv}")
        sys.exit(1)

    # Read the first row of the group CSV to extract the group dimensions
    # (mode, dtype, hdim_q, hdim_v, mask_type). All rows in this CSV share
    # the same group dimensions (they come from the same Step 1 group).
    with in_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows_in = list(reader)
    if not rows_in:
        print(f"[ERROR] Group CSV is empty: {in_csv}")
        sys.exit(1)
    missing = [c for c in GROUP_COLS if c not in rows_in[0]]
    if missing:
        print(
            f"[ERROR] Group CSV is missing columns {missing}; "
            f"please confirm the input is a Step 1 mha_group_*.csv file"
        )
        sys.exit(1)
    group_vals = {c: rows_in[0][c] for c in GROUP_COLS}

    # Flatten --range / --singletons into a sorted set of M values.
    m_values = set()
    for spec in args.range or []:
        m_values.update(_parse_range(spec))
    m_values.update(_parse_singletons(args.singletons))
    if not m_values:
        print(
            "[ERROR] please specify at least one M set via "
            "--range S:E:STEP or --singletons a,b,c"
        )
        sys.exit(1)
    m_values = sorted(m_values)

    # Compute output path: if -o is not provided, replace the input
    # filename prefix 'mha_group_' with 'mha_untune_' to match the Step 1
    # naming convention; otherwise use the user-provided path as-is.
    if args.output:
        out_csv = Path(args.output).absolute()
    else:
        stem = in_csv.name
        if stem.startswith("mha_group_"):
            stem = "mha_untune_" + stem[len("mha_group_") :]
        else:
            stem = "mha_untune_" + stem
        out_csv = in_csv.parent / stem
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    sig = group_signature(tuple(group_vals[c] for c in GROUP_COLS))
    rows_out = []
    for m in m_values:
        r = OrderedDict()
        r["max_seqlen"] = m
        for c in GROUP_COLS:
            r[c] = group_vals[c]
        rows_out.append(r)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["max_seqlen"] + list(GROUP_COLS))
        w.writeheader()
        w.writerows(rows_out)

    print(f"[TUNE] input={in_csv.name}  signature={sig}")
    print(
        f"       max_seqlen count = {len(m_values)}, "
        f"min={m_values[0]}, max={m_values[-1]}"
    )
    print(f"[WRITE] {out_csv}  ({len(rows_out)} rows)")


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        prog="mha_count_shape.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Aggregate MHA forward shape logs (produced by "
            "AITER_DUMP_MHA_FWD_INFO=<stride> at runtime) and turn them "
            "into tune-input CSVs consumable by mha_tune.py.\n"
            "\n"
            "Two-stage workflow:\n"
            "  Stage 1  `group`               : parse log -> per-group CSV +\n"
            "                                   summary CSV + terminal stats.\n"
            "  Stage 2  `generate_tune_range` : pick a `mha_group_*.csv` and\n"
            "                                   emit `mha_untune_*.csv` with\n"
            "                                   the max_seqlen sweep to try.\n"
        ),
        epilog=(
            "End-to-end example:\n"
            "\n"
            "  # 0) Collect a shape log from the running service. `1` means\n"
            "  #    dump every forward call; use e.g. `100` to subsample.\n"
            "  AITER_DUMP_MHA_FWD_INFO=1 \\\n"
            "  AITER_DUMP_MHA_FWD_INFO_FILE=/data1/mha_dump/serviceA.log \\\n"
            "      python your_service.py\n"
            "\n"
            "  # 1) Group by (mode,dtype,hdim_q,hdim_v,mask_type).\n"
            "  python mha_count_shape.py group \\\n"
            "      -i /data1/mha_dump/serviceA.log \\\n"
            "      -d mha_logs/\n"
            "\n"
            "  # 2) For each interesting group, emit a tune sweep.\n"
            "  python mha_count_shape.py generate_tune_range \\\n"
            "      -i mha_logs/mha_group_0_group_bf16_hq72_hv72_mask0.csv \\\n"
            "      --range 512:2048:32 --range 2048:4224:64 \\\n"
            "      --singletons 4096,4104,4144,4176\n"
            "\n"
            "  # 3) Feed the resulting mha_untune_*.csv into mha_tune.py to\n"
            "  #    build/benchmark tile configs and produce mha_tuned_*.csv.\n"
            "\n"
            "Run `mha_count_shape.py <subcommand> -h` for per-subcommand help.\n"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- Stage 1: group ---- #
    p1 = sub.add_parser(
        "group",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="parse the log, group records by "
        "(mode,dtype,hdim_q,hdim_v,mask_type), and emit one "
        "deduplicated shape-combination CSV per group",
        description=(
            "Stage 1. Parse an AITER_DUMP_MHA_FWD_INFO log and split the\n"
            "records into groups keyed by\n"
            "  (mode, dtype, hdim_q, hdim_v, mask_type).\n"
            "\n"
            "For each group, this stage writes ONE CSV whose rows are the\n"
            "unique (seqlens_q, seqlens_k) combinations plus their\n"
            "occurrence count, sorted by count DESC. A cross-group\n"
            "`mha_groups_summary.csv` is also written, and the terminal\n"
            "shows per-group Top-K distributions of max_seqlen_q,\n"
            "seqlens_q (by count and by token weight), and seqlens_k.\n"
            "These stats are meant to help you pick the --range /\n"
            "--singletons for Stage 2."
        ),
        epilog=(
            "Examples:\n"
            "\n"
            "  # Basic: log in cwd, write everything to ./mha_logs/\n"
            "  python mha_count_shape.py group -i mha_fwd.log -d mha_logs/\n"
            "\n"
            "  # Print top-50 seqlens per group (default 20)\n"
            "  python mha_count_shape.py group -i mha_fwd.log -d mha_logs/ \\\n"
            "         --topk 50\n"
            "\n"
            "  # Output files under -d:\n"
            "  #   mha_group_<gid>_<mode>_<dtype>_hq<HQ>_hv<HV>_mask<M>.csv\n"
            "  #   mha_groups_summary.csv\n"
        ),
    )
    p1.add_argument(
        "-i",
        "--input_log",
        required=True,
        metavar="LOG",
        help="path to a plain-text log file that contains `[MHA_FWD] ...` "
        "lines emitted by mha_fwd_dump.h when the process is run with "
        "AITER_DUMP_MHA_FWD_INFO=<stride>. Only one log at a time; "
        "concatenate multiple runs beforehand if needed "
        "(e.g. `cat run1.log run2.log > combined.log`).",
    )
    p1.add_argument(
        "-d",
        "--out_dir",
        default=".",
        metavar="DIR",
        help="output directory for the per-group CSVs and "
        "`mha_groups_summary.csv` (default: %(default)s). Created "
        "if it does not exist.",
    )
    p1.add_argument(
        "--topk",
        type=int,
        default=20,
        metavar="K",
        help="how many top entries to print for each per-group "
        "distribution (max_seqlen_q, seqlens_q by call-count, "
        "seqlens_q by token weight, and the same two for seqlens_k) "
        "(default: %(default)s). Only affects terminal output; CSVs "
        "always contain the full unique set.",
    )
    p1.set_defaults(func=cmd_group)

    # ---- Stage 2: generate_tune_range ---- #
    p2 = sub.add_parser(
        "generate_tune_range",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="read one mha_group_*.csv produced by Stage 1 and generate "
        "the corresponding mha_untune_*.csv driven by --range / "
        "--singletons",
        description=(
            "Stage 2. Take one Stage-1 group CSV and expand a\n"
            "user-specified `max_seqlen` sweep into an untuned CSV that\n"
            "mha_tune.py can consume. Every output row inherits the\n"
            "(mode, dtype, hdim_q, hdim_v, mask_type) of the input group\n"
            "and adds one `max_seqlen` value to try. The final sweep is\n"
            "the sorted UNION of all --range segments and --singletons\n"
            "(duplicates are dropped)."
        ),
        epilog=(
            "Examples:\n"
            "\n"
            "  # Fine step below 2048, coarser above, plus a few hot\n"
            "  # discrete lengths observed in Stage-1 terminal stats.\n"
            "  python mha_count_shape.py generate_tune_range \\\n"
            "      -i mha_logs/mha_group_0_group_bf16_hq72_hv72_mask0.csv \\\n"
            "      --range 512:2048:32 --range 2048:4224:64 \\\n"
            "      --singletons 4096,4104,4144,4176\n"
            "\n"
            "  # Custom output path\n"
            "  python mha_count_shape.py generate_tune_range \\\n"
            "      -i mha_logs/mha_group_1_group_bf16_hq256_hv256_mask2.csv \\\n"
            "      -o mha_logs/hq256_untune.csv \\\n"
            "      --range 256:2560:128\n"
            "\n"
            "  # Singletons only (no arithmetic range at all)\n"
            "  python mha_count_shape.py generate_tune_range \\\n"
            "      -i mha_logs/mha_group_2_group_fp16_hq128_hv128_mask0.csv \\\n"
            "      --singletons 512,1024,2048,4096\n"
        ),
    )
    p2.add_argument(
        "-i",
        "--input_csv",
        required=True,
        metavar="CSV",
        help="group CSV produced by Stage 1, i.e. "
        "`mha_group_<gid>_<mode>_<dtype>_hq<HQ>_hv<HV>_mask<M>.csv`. "
        "Only the first row is inspected to lift "
        "(mode,dtype,hdim_q,hdim_v,mask_type); every row in a Stage-1 "
        "CSV already shares those columns.",
    )
    p2.add_argument(
        "-o",
        "--output",
        default="",
        metavar="CSV",
        help="output CSV path. When omitted (default), it is placed next "
        "to the input with the filename prefix rewritten "
        "`mha_group_*` -> `mha_untune_*` (matches Stage-1 naming so "
        "mha_tune.py picks it up unchanged).",
    )
    p2.add_argument(
        "--range",
        action="append",
        default=[],
        metavar="S:E:STEP",
        help="closed-interval arithmetic range of max_seqlen values, i.e. "
        "S, S+STEP, S+2*STEP, ..., up to and including E. S/E/STEP "
        "must be positive integers with STEP>0 and E>=S. May be "
        "given MULTIPLE times to concatenate segments with different "
        "step sizes (e.g. dense sweep near hot region + sparse "
        "sweep on tail).",
    )
    p2.add_argument(
        "--singletons",
        default="",
        metavar="N1,N2,...",
        help="extra discrete max_seqlen values, comma-separated "
        "(e.g. `480,1600,4176`). Useful for exact-observed lengths "
        "from Stage-1 stats that fall off the --range grid. Merged "
        "with --range via set union then sorted ascending.",
    )
    p2.set_defaults(func=cmd_generate_tune_range)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
