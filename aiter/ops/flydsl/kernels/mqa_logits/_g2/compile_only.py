# SPDX-License-Identifier: MIT
"""Compile-only check for the v0 kernel: NO device, NO kernel launch.

Runs in a container started without ``--device=/dev/kfd`` / ``--device=/dev/dri``,
so it cannot touch the GPU. It exercises the whole FlyDSL lowering path for both
dtype specializations (bf16 and fp8) and reports whether a code object was
produced, plus its sha256.

    COMPILE_ONLY=1 python3 compile_only.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys

os.environ.setdefault("COMPILE_ONLY", "1")

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from _load import load_kernel_module  # noqa: E402


def main() -> int:
    K, how = load_kernel_module()
    print("kernel module:", how)
    rows, columns, pages, reqs, width = 4, 64, 8, 1, 16
    q = torch.zeros(rows * 4 * 128, dtype=torch.uint8)
    kv = torch.zeros(pages * 4 * 128, dtype=torch.uint8)
    ks = torch.zeros(pages * 4, dtype=torch.float32)
    pt = torch.zeros(reqs * width, dtype=torch.int32)
    t2r = torch.zeros(rows, dtype=torch.int32)
    qp = torch.zeros(rows, dtype=torch.int32)
    cl = torch.zeros(reqs, dtype=torch.int32)
    logits = torch.zeros(rows * columns, dtype=torch.float32)
    vis = torch.zeros(rows, dtype=torch.int32)
    rst = torch.zeros(rows, dtype=torch.int32)
    grid_x = (columns + K.THREADS - 1) // K.THREADS
    out = {"module": how, "specializations": []}
    rc = 0
    for is_fp8 in (False, True):
        tag = "fp8" if is_fp8 else "bf16"
        rec = {"dtype": tag}
        try:
            launcher = K._build_kernel(is_fp8, K.THREADS, math.sqrt(128))
            cf = flyc.compile(
                launcher,
                q, kv, ks, pt, t2r, qp, cl, logits, vis, rst,
                grid_x, rows, columns, pages, reqs, width, fx.Stream(None),
            )
            rec["compiled"] = True
            rec["compiled_function"] = type(cf).__name__
            print(f"[COMPILED] {tag} kernel={launcher.kernel_name}")
        except Exception as exc:  # noqa: BLE001
            rec["compiled"] = False
            rec["error"] = f"{type(exc).__name__}: {exc}"[:6000]
            rc = 1
            print(f"[FAILED]   {tag}: {rec['error'][:2000]}")
        out["specializations"].append(rec)
    root = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "")
    hits = []
    for dp, _, fs in os.walk(root) if root else []:
        for f in sorted(fs):
            if f.endswith((".hsaco", ".co")):
                p = os.path.join(dp, f)
                with open(p, "rb") as fh:
                    hits.append({"path": p, "bytes": os.path.getsize(p),
                                 "sha256": hashlib.sha256(fh.read()).hexdigest()})
    out["code_objects"] = hits
    print(json.dumps(out, indent=2))
    dest = os.environ.get("QSA_RESULTS", ".")
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "compile_only.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return rc


if __name__ == "__main__":
    sys.exit(main())
