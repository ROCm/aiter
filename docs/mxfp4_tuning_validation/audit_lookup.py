# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import aiter.ops.gemm_op_a4w4 as op
from aiter.jit.core import AITER_CONFIG
from aiter.utility import mxfp4_tuning as api


def rows(path):
    frame = pd.read_csv(path)
    return {
        tuple(r[c] for c in ("gfx", "cu_num", "M", "N", "K")): (
            r["kernelName"],
            int(r["splitK"]),
        )
        for _, r in frame.iterrows()
    }


parser = argparse.ArgumentParser()
parser.add_argument("--baseline", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
oldfile = args.baseline
output = Path(args.output_dir)
output.mkdir(parents=True, exist_ok=True)
old = rows(oldfile)
sources = api._default_sources()
newfile = api.get_mxfp4_config_file()
new = rows(newfile)
assert old == new, (len(old), len(new))
queries = set(old)
profile_keys = {}
for profile in api.available_mxfp4_profiles():
    keys = set()
    for path in api._profile_sources(profile):
        keys.update(rows(path))
    profile_keys[profile] = keys
    queries.update(keys)
for gfx, cu, m, n, k in list(queries):
    for delta in (-31, -1, 1):
        if m + delta > 0:
            queries.add((gfx, cu, m + delta, n, k))
queries = sorted(queries)


def lookup(path):
    result = {}
    with patch.object(
        AITER_CONFIG, "AITER_CONFIG_GEMM_A4W4_FILE", property(lambda _: str(path))
    ):
        for gfx, cu in sorted({q[:2] for q in queries}):
            op.get_GEMM_config.cache_clear()
            for attr in ("gemm_dict", "has_gfx"):
                if hasattr(op.get_GEMM_config, attr):
                    delattr(op.get_GEMM_config, attr)
            with patch.object(op, "get_gfx", return_value=gfx), patch.object(
                op, "get_cu_num", return_value=cu
            ):
                for key in queries:
                    if key[:2] != (gfx, cu):
                        continue
                    cfg = op.get_GEMM_config(*key[2:])
                    result[key] = (
                        None if cfg is None else (cfg["kernelName"], int(cfg["splitK"]))
                    )
    return result


baseline = lookup(oldfile)
current = lookup(newfile)
assert baseline == current
print(
    json.dumps(
        {
            "default_keys": len(new),
            "lookup_queries": len(queries),
            "default_lookup_differences": 0,
        }
    ),
    flush=True,
)
for profile, keys in profile_keys.items():
    file = output / (profile + "-effective.csv")
    file.write_bytes(api._merged_csv(api.resolve_mxfp4_tuning_sources(profile=profile)))
    selected = lookup(file)
    changed = [q for q in queries if baseline[q] != selected[q]]
    unrelated = []
    for q in changed:
        gfx, cu, m, n, k = q
        candidates = {
            (gfx, cu, x, n, k)
            for x in (m, op.get_padded_m(m, n, k, 0), op.get_padded_m(m, n, k, 1))
        }
        if not candidates & keys:
            unrelated.append(q)
    assert not unrelated, unrelated[:5]
    record = {
        "profile": profile,
        "profile_keys": len(keys),
        "effective_keys": len(rows(file)),
        "lookup_differences": len(changed),
        "unrelated_differences": len(unrelated),
        "changes": [
            {
                "key": q,
                "old": baseline[q],
                "new": selected[q],
                "exact_profile_key": q in keys,
            }
            for q in changed
        ],
    }
    (output / (profile + "-lookup.json")).write_text(json.dumps(record, indent=2))
    print(json.dumps({k: v for k, v in record.items() if k != "changes"}), flush=True)
