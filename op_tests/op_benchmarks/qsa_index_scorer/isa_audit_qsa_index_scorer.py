# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ISA audit of the QSA index scorer API (no device: compile only, stock FlyDSL, gfx1250).

For every profile row (default: verified rows) and both metadata forms (int64 /
int32 query_positions) this compiles, in fresh no-device processes,

* ``api``  -- the specialization the public op ``qsa_paged_mqa_logits(profile=...)``
              dispatches for the row's shape (real wrapper, real router, real
              builder + compiler; only ``_run`` and the stream lookup intercepted);
* ``base`` -- the same shape through the champion kernel module directly
              (``--base-source``: the profiled champion ``82c4dec5``; with
              ``--cache-dtype bf16 --base-source <a78c285c kernel>`` it proves the
              bf16 ISA identity that carries the a78c285c Phase-1 evidence),

and checks: identical normalized final ISA (``21_final_isa.s`` minus .loc/.file/
comments) api vs base, exactly one launch per call, zero scratch
(``private_segment_fixed_size == 0``), and records VGPR/SGPR/LDS and mnemonic counts.

    COMPILE_ONLY=1 FLYDSL_GPU_ARCH=gfx1250 python3 -B isa_audit_qsa_index_scorer.py \
        --base-source /base/<kernel path> --results /results/isa [--all-rows] [--jobs 16]

Requires no /dev/kfd and no /dev/dri (asserted).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "aiter" / "ops" / "qsa_index_scorer"
KERNEL = ROOT / "aiter" / "ops" / "flydsl" / "kernels" / "mqa_logits" / \
    "qsa_paged_mqa_logits_page4_gfx1250.py"
PROFILE = ROOT / "aiter" / "configs" / "qsa_index_scorer" / "qsa_page4_gfx1250_v1.json"


def worker(a) -> int:
    assert not Path("/dev/kfd").exists() and not Path("/dev/dri").exists(), "devices visible"
    row = json.loads(a.row)
    dest = Path(a.results) / f"{row['cell']}-{a.meta}-{a.side}"
    dest.mkdir(parents=True, exist_ok=True)
    os.environ.update(COMPILE_ONLY="1", FLYDSL_GPU_ARCH="gfx1250", FLYDSL_DUMP_IR="1",
                      FLYDSL_DEBUG_DUMP_ASM="1", FLYDSL_DUMP_DIR=str(dest / "isa"),
                      FLYDSL_RUNTIME_CACHE_DIR=str(dest / "cache"))
    import torch
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    def forbidden(*args, **kwargs):
        raise AssertionError("GPU discovery/runtime forbidden in the no-device ISA audit")

    torch.cuda.current_stream = lambda *args, **kwargs: None
    for name in ("is_available", "device_count", "init", "_lazy_init", "synchronize"):
        setattr(torch.cuda, name, forbidden)
    torch.set_num_threads(1)
    src = Path(a.source)
    spec = importlib.util.spec_from_file_location(f"qsa_isa_{a.side}", src)
    K = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(K)

    rows, reqs = row["rows"]["min"], row["requests"]["min"]
    width, cols, fp8 = row["table_width"], row["columns"], row["cache_dtype"] == "fp8"
    q = torch.zeros((rows, 4, 128), dtype=torch.bfloat16)
    size = 8 * 4 * 128 * (1 if fp8 else 2)
    back = torch.zeros(size + 128, dtype=torch.uint8)
    off = (-back.data_ptr()) % 128
    cache = back[off:off + size].view(torch.float8_e4m3fn if fp8 else torch.bfloat16).view(8, 4, 1, 128)
    scale = torch.ones((8, 4, 1), dtype=torch.float32) if fp8 else None
    table = torch.zeros((reqs, width), dtype=torch.int32)
    t2r = (torch.arange(rows, dtype=torch.int32) * reqs) // rows if rows > reqs else \
        torch.arange(rows, dtype=torch.int32)
    qp = torch.full((rows,), cols * 4 - 1, dtype=torch.int64 if a.meta == "qp64" else torch.int32)
    ctx = torch.full((reqs,), cols * 4, dtype=torch.int32)
    logits = torch.empty((rows, cols), dtype=torch.float32)
    vis = torch.empty(rows, dtype=torch.int32)
    rst = torch.empty(rows, dtype=torch.int32)
    caps = []

    def compile_only(launch, *args):
        tensors = [t for t in args if isinstance(t, torch.Tensor)]
        assert all(t.device.type == "cpu" for t in tensors)
        flyc.compile(launch, *args[:-1], fx.Stream(None))
        caps.append({"kernel_name": launch.kernel_name,
                     "scalars": [x for x in args[:-1] if not isinstance(x, torch.Tensor)]})

    K._run = compile_only
    rec = {"cell": row["cell"], "vid": row["vid"], "side": a.side, "meta": a.meta,
           "source": str(src), "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
           "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("QSA_")}}
    if a.side == "api":
        pspec = importlib.util.spec_from_file_location(
            "qsa_isc_isa", PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)])
        pkg = importlib.util.module_from_spec(pspec)
        sys.modules["qsa_isc_isa"] = pkg
        pspec.loader.exec_module(pkg)
        api = sys.modules["qsa_isc_isa.api"]
        api._ALLOWED_DEVICE_TYPES = ("cuda", "cpu")
        api._device_info = lambda dev: ("gfx1250", 256)
        api._KERNEL, api._KERNEL_ENV = K, api.P.effective_kernel_env()
        d = api.decide(q, cache, table, t2r, qp, ctx, 4, None, cols, rst, False, scale,
                       logits=logits, visible_groups=vis, profile=str(a.profile),
                       allow_unverified=True)
        rec["decision"] = {"ok": d.ok, "reason": d.reason.value, "vid": d.vid, "cell": d.cell}
        assert d.ok, rec["decision"]
        if row["golden"]["status"] == "pass":    # the public op, exactly as callers use it
            api.qsa_paged_mqa_logits(q, cache, table, t2r, qp, ctx, 4, None, cols, rst, False,
                                     scale, logits=logits, visible_groups=vis,
                                     profile=str(a.profile))
        else:                                    # --all-rows: pending rows (sweep path)
            api.try_qsa_paged_mqa_logits(q, cache, table, t2r, qp, ctx, 4, None, cols, rst,
                                         False, scale, logits=logits, visible_groups=vis,
                                         profile=str(a.profile), allow_unverified=True)
    else:
        K.qsa_paged_mqa_logits_page4(q, cache, table, t2r, qp, ctx, compress_ratio=4,
                                     max_columns=cols, compressed_k_scale=scale, logits=logits,
                                     visible_groups=vis, row_starts=rst)
    assert len(caps) == 1, caps
    rec.update(caps[0])
    paths = sorted((dest / "isa").glob("*/21_final_isa.s"))
    assert len(paths) == 1, paths
    text = paths[0].read_text()
    body = [ln.strip() for ln in text.splitlines()]
    normal = "\n".join(ln for ln in body if ln and not ln.startswith((".loc", ".file", ";", "//")))
    instr = [ln for ln in body if ln and re.match(r"^[a-z_][a-z0-9_.]*(\s|$)", ln)
             and not ln.startswith(".")]
    mn: dict = {}
    for ln in instr:
        mn[ln.split()[0]] = mn.get(ln.split()[0], 0) + 1
    rec["isa"] = {"path": str(paths[0]), "sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
                  "normalized_sha256": hashlib.sha256(normal.encode()).hexdigest(),
                  "instructions": len(instr), "mnemonics": mn}
    rec["resources"] = {k: int(re.search(r"\.amdhsa_" + k + r"\s+(\d+)", text)[1])
                        for k in ("next_free_vgpr", "next_free_sgpr", "group_segment_fixed_size",
                                  "private_segment_fixed_size")}
    (dest / "audit.json").write_text(json.dumps(rec, indent=2) + "\n")
    print("QSA_ISA_AUDIT_PASS", a.side, a.meta, row["cell"], rec["kernel_name"],
          rec["isa"]["normalized_sha256"], flush=True)
    return 0


def driver(a) -> int:
    assert not Path("/dev/kfd").exists() and not Path("/dev/dri").exists(), "devices visible"
    prof = json.loads(Path(a.profile).read_text())
    rows = [r for r in prof["rows"] if (a.all_rows or r["golden"]["status"] == "pass")
            and (not a.cache_dtype or r["cache_dtype"] == a.cache_dtype)]
    out = Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for r in rows:
        for meta in ("qp64", "i32"):
            for side, src in (("api", str(KERNEL)), ("base", a.base_source)):
                jobs.append((r, meta, side, src))

    def run(job):
        r, meta, side, src = job
        cmd = [sys.executable, "-B", __file__, "--worker", "--side", side, "--source", src,
               "--meta", meta, "--row", json.dumps(r), "--results", str(out),
               "--profile", str(a.profile)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        log = out / f"{r['cell']}-{meta}-{side}.log"
        log.write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
        return job, p.returncode

    fails = []
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for (r, meta, side, _), rc in ex.map(run, jobs):
            if rc:
                fails.append(f"{r['cell']}-{meta}-{side} rc={rc}")
    summary = {"rows": len(rows), "cache_dtype": a.cache_dtype or "all", "all_rows": bool(a.all_rows),
               "compiles": len(jobs), "compile_failures": fails, "pairs": []}
    for r in rows:
        for meta in ("qp64", "i32"):
            pair = {}
            for side in ("api", "base"):
                f = out / f"{r['cell']}-{meta}-{side}" / "audit.json"
                pair[side] = json.loads(f.read_text()) if f.exists() else None
            api_, base = pair["api"], pair["base"]
            ok = bool(api_ and base and api_["isa"]["normalized_sha256"] == base["isa"]["normalized_sha256"]
                      and api_["kernel_name"] == base["kernel_name"]
                      and api_["resources"] == base["resources"]
                      and api_["resources"]["private_segment_fixed_size"] == 0)
            summary["pairs"].append({
                "cell": r["cell"], "vid": r["vid"], "meta": meta, "identical": ok,
                "kernel_name": api_ and api_["kernel_name"],
                "normalized_sha256": api_ and api_["isa"]["normalized_sha256"],
                "base_normalized_sha256": base and base["isa"]["normalized_sha256"],
                "resources": api_ and api_["resources"]})
    summary["all_identical"] = (not fails) and all(p["identical"] for p in summary["pairs"])
    summary["source_sha256"] = hashlib.sha256(KERNEL.read_bytes()).hexdigest()
    summary["base_sha256"] = hashlib.sha256(Path(a.base_source).read_bytes()).hexdigest()
    (out / "isa-audit-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for p in summary["pairs"]:
        print(("IDENTICAL " if p["identical"] else "DIFFERENT ") + f"{p['cell']} {p['meta']} "
              f"{p['kernel_name']} {p['resources']}")
    print("ISA_AUDIT", "PASS" if summary["all_identical"] else "FAIL",
          f"pairs={len(summary['pairs'])} failures={fails}")
    return 0 if summary["all_identical"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--side", choices=("api", "base"))
    ap.add_argument("--source")
    ap.add_argument("--meta", choices=("qp64", "i32"), default="qp64")
    ap.add_argument("--row")
    ap.add_argument("--results", required=True)
    ap.add_argument("--profile", default=str(PROFILE))
    ap.add_argument("--base-source")
    ap.add_argument("--all-rows", action="store_true")
    ap.add_argument("--cache-dtype", choices=("bf16", "fp8"), help="audit only rows of this dtype")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("ISA_JOBS", "8")))
    a = ap.parse_args(argv)
    if a.worker:
        return worker(a)
    if not a.base_source:
        ap.error("--base-source is required (champion kernel file)")
    return driver(a)


if __name__ == "__main__":
    sys.exit(main())
