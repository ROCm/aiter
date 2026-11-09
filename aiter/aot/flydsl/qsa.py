#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation of the fused QSA index scorer (gfx1250).

``aiter.ops.qsa.qsa_paged_mqa_logits`` serves a call class with a FlyDSL gfx1250
kernel when the active profile (``aiter/configs/qsa_index_scorer/<name>.json``) covers
it; the kernel, its specialization (page size 4 / 8 / 16, bf16 / fp8 cache, route,
launch knobs) and its launch geometry are picked on the host from the class's shape.
This module restates none of that. It turns every call class the shipped profile can
serve -- each verified row's exact shape, each verified rule's sample classes and each
verified bucket's sample classes, at their page size and cache dtype, with int64
query positions -- into a job, and :mod:`.replay` calls the neutral op itself on
FakeTensors inside :func:`aiter.aot.flydsl.dry_run.dry_run` (``COMPILE_ONLY=1``, GPU
runtime forbidden) with the profile configured (``aiter.ops.qsa.configure``). Every
FlyDSL launch the op reaches is compiled into the cache instead of run. A replayed
class that the op does not serve fused fails the job (the profile claims it); a
compile error fails it too (the op's first-launch native fallback is off here).

``--check`` (``FLYDSL_RUNTIME_RUN_ONLY=1``) compiles nothing and fails on any kernel
the replay reaches that the cache lacks, i.e. it proves the runtime's cache keys hit
the artifacts the build wrote. The keys do not depend on the order variants compile
in within a process (kernel 6b537f3b; see ``_raw`` in the kernel module), so the
artifacts of a parallel AOT build are the ones a serving process looks up.

Measured without a device (no-device container, 256 cores, flydsl 0.4.0 -- this
driver's replay hooks need flydsl >= 0.4 --, profile ``qsa_page4_gfx1250_v1`` at
kernel 6b537f3b): 2578 call classes replayed in 162 worker batches, 39 kernels
compiled (one per profile variant x page size x cache dtype, int64 positions;
808 KiB of cache), cold 9.2 s (11.9 s wall with process start-up); ``--check`` 4.4 s
with every launch a cache hit; a second build adds no artifact; ``--check`` against an
empty cache fails all 2578 replays.

Usage:
    python -m aiter.aot.flydsl.qsa                        # shipped profile
    python -m aiter.aot.flydsl.qsa --profile /path/x.json
    python -m aiter.aot.flydsl.qsa --check                # cache-hit check, compiles nothing
"""

from __future__ import annotations

import argparse
import json
import math
import os

LABEL = "QSA"
# No tuned CSV: the jobs come from the scorer profile.
DEFAULT_CSVS: list[str] = []
AOT_ARCH_DEFAULT = "gfx1250"
DEFAULT_PROFILE = "qsa_page4_gfx1250_v1"
_CONFIG_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "configs",
        "qsa_index_scorer",
    )
)
_HEADS, _DIM, _RATIO = 4, 128, 4
_SOURCE = "aiter.aot.flydsl.qsa"


def profile_path(name_or_path: str | None = None) -> str:
    """A profile path from a bare name (``aiter/configs/qsa_index_scorer/<name>.json``)
    or a path."""
    name = name_or_path or DEFAULT_PROFILE
    if os.sep in name or "/" in name or name.endswith(".json"):
        return os.path.abspath(name)
    return os.path.join(_CONFIG_DIR, f"{name}.json")


def _page(rec: dict) -> int:
    return int(rec.get("page_size", 4))


def _passed(rec: dict) -> bool:
    return (rec.get("golden") or {}).get("status") == "pass"


def jobs_for_profile(path: str) -> list[dict]:
    """One job per distinct call class the profile can serve (verified records only:
    pending / failed records are never selected at runtime)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    jobs, seen = [], set()

    def add(rec, kind, shape, fixture=None):
        page = _page(rec)
        cls = (
            rec["cache_dtype"],
            page,
            int(shape["rows"]),
            int(shape["requests"]),
            int(shape["columns"]),
            int(shape["table_width"]),
            shape.get("layout"),
        )
        if cls in seen:
            return
        seen.add(cls)
        jobs.append(
            {
                "kernel_name": f"{rec['vid']}/{rec['cache_dtype']}",
                "gfx": rec.get("arch") or AOT_ARCH_DEFAULT,
                "cu_num": int(rec.get("cu") or 256),
                "profile": path,
                "kind": kind,
                "id": rec.get("cell") or rec.get("id"),
                "cache_dtype": rec["cache_dtype"],
                "page": page,
                "rows": cls[2],
                "requests": cls[3],
                "columns": cls[4],
                "table_width": cls[5],
                "layout": cls[6],
                "pos_dtype": "int64",
                "fixture": fixture,
            }
        )

    for r in data.get("rows") or []:
        if _passed(r):
            shape = {
                "rows": r["rows"]["min"],
                "requests": r["requests"]["min"],
                "columns": r["columns"],
                "table_width": r["table_width"],
            }
            add(r, "row", shape, r.get("fixture") or None)
    for kind in ("rules", "buckets"):
        for r in data.get(kind) or []:
            if _passed(r):
                for s in r.get("samples") or []:
                    add(r, kind[:-1], s)
    return jobs


def describe(job: dict) -> str:
    return (
        f"{job['kind']} {job['id']} P{job['page']} {job['cache_dtype']} "
        f"r{job['rows']}q{job['requests']} c{job['columns']} w{job['table_width']}"
        + (f" {job['layout']}" if job.get("layout") else "")
    )


def _aligned(shape, dtype, align: int):
    """A (Fake)Tensor on the logical GPU whose ``data_ptr`` is ``align``-aligned, as the
    callers' caches are (the router keys on the cache base alignment)."""
    import torch

    n = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    raw = torch.empty(n + align, dtype=torch.uint8, device="cuda")
    off = (-raw.data_ptr()) % align
    return raw[off : off + n].view(dtype).view(*shape)


def prepare(_jobs: list[dict]) -> None:
    """Import the op and the kernel module in the per-arch process, before any
    :func:`~aiter.aot.flydsl.dry_run.dry_run`: the kernel module binds
    ``torch.cuda.current_stream`` at import to pick its raw-stream fast path, which
    must be the real function, not a dry run's stand-in (else a replay that imports
    the module inside a dry run would query the GPU runtime for the stream)."""
    from aiter.ops import qsa  # noqa: F401
    from aiter.ops.qsa_index_scorer import dispatch

    dispatch._api()._load_kernel()


def replay(job: dict) -> None:
    import torch

    from aiter.ops import qsa
    from aiter.ops.qsa_index_scorer import dispatch, gen_profile

    dispatch.FUSED_FALLBACK_ON_ERROR = False  # a compile error fails the build
    if qsa.status()["profile"] != job["profile"]:
        qsa.configure(profile=job["profile"], source=_SOURCE)
    rows, reqs = job["rows"], job["requests"]
    cols, width, page = job["columns"], job["table_width"], job["page"]
    fx = job.get("fixture") or gen_profile.sample_fixture(
        {
            "rows": rows,
            "requests": reqs,
            "table_width": width,
            "page_size": page,
            "layout": job.get("layout"),
        }
    )
    pages = int(fx.get("cache_pages") or gen_profile.E2E_PAGES)
    fp8 = job["cache_dtype"] == "fp8"
    q = _aligned((rows, _HEADS, _DIM), torch.bfloat16, 16)
    cache = _aligned(
        (pages, page, 1, _DIM), torch.float8_e4m3fn if fp8 else torch.bfloat16, 256
    )
    scale = (
        torch.empty((pages, page, 1), dtype=torch.float32, device="cuda")
        if fp8
        else None
    )
    pt = torch.empty((reqs, width), dtype=torch.int32, device="cuda")
    t2r = torch.empty(rows, dtype=torch.int32, device="cuda")
    pos = torch.empty(rows, dtype=getattr(torch, job["pos_dtype"]), device="cuda")
    ctx = torch.empty(reqs, dtype=torch.int32, device="cuda")
    rs = torch.empty(rows, dtype=torch.int32, device="cuda")
    args = (q, cache, pt, t2r, pos, ctx, _RATIO, None, cols, rs, False, scale)
    d = qsa.explain(*args)
    if d.path != "fused":
        raise RuntimeError(
            f"{describe(job)}: the profile covers it, the op serves it {d.path} "
            f"({d.reason}: {d.detail})"
        )
    qsa.qsa_paged_mqa_logits(*args)


def collect_jobs(_csv_paths: list[str], args: argparse.Namespace | None) -> list:
    profile = getattr(args, "profile", None) if args is not None else None
    return jobs_for_profile(profile_path(profile))


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        default=None,
        help=f"scorer profile name or path (default {DEFAULT_PROFILE})",
    )


def main() -> None:
    from aiter.aot.flydsl import replay as _replay

    _replay.main(
        "aiter.aot.flydsl.qsa",
        "AOT pre-compile the fused QSA index scorer (gfx1250) for every call class "
        "its profile serves",
    )


if __name__ == "__main__":
    main()
