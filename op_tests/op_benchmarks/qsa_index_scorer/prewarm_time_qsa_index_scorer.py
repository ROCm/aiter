# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""R2 warm-cache timing of ``prewarm`` (per-variant seconds, cold vs warm restart).

Two modes:

* ``--compile-only`` (no device; CPU dummy tensors, ``flyc.compile`` only): measures the
  FlyDSL build + LLVM compile seconds of every variant and populates the persistent
  cache. Run twice with the same ``--cache-dir`` in two processes: the second run is the
  warm restart (disk-cache hit). Needs no /dev/kfd, no /dev/dri::

      COMPILE_ONLY=1 FLYDSL_GPU_ARCH=gfx1250 python3 prewarm_time_qsa_index_scorer.py \
          --compile-only --cache-dir /w/cache --page-sizes 4 --cache-dtypes bf16 --pos i64 \
          [--jobs 4] [--include-pending] --json /w/cold.json

* GPU (default): the real ``prewarm`` (build + compile + one private launch per variant),
  under the node's gpu-lock, same flags (``--device cuda``).

``--include-pending`` treats pending rows/rules as selectable (timing of page-8/16
variants before their GPU check; never used for serving).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)


def _parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument(
        "--cache-dir", help="persistent FlyDSL cache dir (default: the op's default)"
    )
    ap.add_argument(
        "--page-sizes", default="", help="comma list, e.g. 4 or 4,16 (default all)"
    )
    ap.add_argument("--cache-dtypes", default="bf16")
    ap.add_argument("--pos", default="i64", choices=("i32", "i64", "both"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--include-pending", action="store_true")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = _parse(argv)
    if a.compile_only:
        os.environ.setdefault("COMPILE_ONLY", "1")
        os.environ.setdefault("FLYDSL_GPU_ARCH", "gfx1250")
    t_import = time.perf_counter()
    import torch

    if a.compile_only:
        torch.cuda.current_stream = lambda *x, **k: None
    d = os.path.join(ROOT, "aiter", "ops", "qsa_index_scorer")
    spec = importlib.util.spec_from_file_location(
        "qsa_isc_pw", os.path.join(d, "__init__.py"), submodule_search_locations=[d]
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["qsa_isc_pw"] = pkg
    spec.loader.exec_module(pkg)
    api = sys.modules["qsa_isc_pw.api"]
    P = api.P
    if a.include_pending:
        P.ProfileRule.verified = property(lambda self: True)
        P.ProfileRow.verified = property(lambda self: True)
        P.ProfileBucket.verified = property(
            lambda self: True
        )  # R2E: knob-pinned buckets
    t0 = time.perf_counter()
    K = api._load_kernel()
    t_kernel = time.perf_counter() - t0
    pos = {
        "i32": (torch.int32,),
        "i64": (torch.int64,),
        "both": (torch.int32, torch.int64),
    }[a.pos]
    pages = tuple(int(x) for x in a.page_sizes.split(",") if x) or None
    profile = a.profile or pkg.DEFAULT_PROFILE
    plan = api.prewarm_plan(
        profile,
        cache_dtypes=tuple(a.cache_dtypes.split(",")),
        pos_dtypes=pos,
        page_sizes=pages,
        kernel=K,
    )
    t0 = time.perf_counter()
    tags = api.prewarm(
        profile,
        cache_dtypes=tuple(a.cache_dtypes.split(",")),
        pos_dtypes=pos,
        page_sizes=pages,
        cache_dir=a.cache_dir,
        jobs=a.jobs,
        compile_only=a.compile_only,
        device=None if a.compile_only else a.device,
    )
    total = time.perf_counter() - t0
    cdir = a.cache_dir or api.compile_cache_dir()
    n_pkl = (
        sum(f.endswith(".pkl") for _r, _d, fs in os.walk(cdir) for f in fs)
        if cdir and os.path.isdir(cdir)
        else None
    )
    rec = {
        "mode": "compile_only" if a.compile_only else "gpu",
        "jobs": a.jobs,
        "kernel_page_sizes": list(api.kernel_page_sizes(K)),
        "planned": [p[0] for p in plan],
        "built": tags,
        "total_s": round(total, 3),
        "kernel_import_s": round(t_kernel, 3),
        "process_s": round(time.perf_counter() - t_import, 3),
        "per_variant_s": {
            k: round(v["seconds"], 3) for k, v in api.prewarm_stats().items()
        },
        "cache_dir": cdir,
        "cache_pkl_files": n_pkl,
        "flydsl_cache_env_after": os.environ.get("FLYDSL_RUNTIME_CACHE_DIR"),
    }
    print(json.dumps(rec, indent=1))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rec, fh, indent=1)
    # knob-pinned specializations (e.g. decg + group_rows 3) share their variant tag
    return 0 if tags == sorted({p[0] for p in plan}) else 1


if __name__ == "__main__":
    sys.exit(main())
