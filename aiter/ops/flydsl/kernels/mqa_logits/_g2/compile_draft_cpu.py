# SPDX-License-Identifier: MIT
"""Round DRAFT: stock FlyDSL no-device compile of every draft specialization.

NO device, NO launch (COMPILE_ONLY=1). With FLYDSL_DUMP_IR=1
FLYDSL_DEBUG_DUMP_ASM=1 each kernel's final ISA lands in
``$FLYDSL_DUMP_DIR/<kernel_name>/21_final_isa.s``.
"""

from __future__ import annotations

import json
import math
import os
import sys

os.environ.setdefault("COMPILE_ONLY", "1")

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _load import load_kernel_module  # noqa: E402


def main() -> int:
    K, how = load_kernel_module()
    rows, columns, pages, reqs, width = 16, 1024, 512, 16, 256
    rc, out = 0, []
    for is_fp8 in (False, True):
        for k_load, wc in (("tdm", 32), ("tdm", 64), ("raw", 64), ("raw", 32)):
            tag = f"{'fp8' if is_fp8 else 'bf16'}_{k_load}_w{wc}"
            rec = {"variant": tag}
            try:
                launch = K._build_draft_kernel(is_fp8, math.sqrt(128), k_load, wc)
                gx, gy = K.draft_grid(rows, columns, wc)
                eb = 1 if is_fp8 else 2
                args = (torch.zeros(rows * 1024, dtype=torch.uint8),
                        torch.zeros(pages * 512 * eb, dtype=torch.uint8),
                        torch.zeros(pages * 4, dtype=torch.float32),
                        torch.zeros(reqs * width, dtype=torch.int32),
                        torch.zeros(rows, dtype=torch.int32),
                        torch.zeros(rows, dtype=torch.int32),
                        torch.zeros(reqs, dtype=torch.int32),
                        torch.zeros(rows * columns, dtype=torch.float32),
                        torch.zeros(rows, dtype=torch.int32),
                        torch.zeros(rows, dtype=torch.int32))
                flyc.compile(launch, *args, gx, gy, rows, columns, pages, reqs, width,
                             fx.Stream(None))
                rec.update(compiled=True, kernel_name=launch.kernel_name, grid=[gx, gy],
                           cols_per_block=launch.cols_per_block)
                print(f"[COMPILED] {tag} kernel={launch.kernel_name}")
            except Exception as exc:  # noqa: BLE001
                rc = 1
                rec.update(compiled=False, error=f"{type(exc).__name__}: {exc}"[:6000])
                print(f"[FAILED]   {tag}: {rec['error'][:3000]}")
            out.append(rec)
    if K.TDM_R16_NOBAR:  # QSA_R16_NOBAR=1 at import: the barrier-free bf16 r16 TDM kernel
        rec = {"variant": "bf16_r16_tdm_nobar"}
        try:
            launch = K._build_wmma_kernel(math.sqrt(128), 16, "tdm")
            assert launch.kernel_name.endswith("_tdm_nobar"), launch.kernel_name
            args = (torch.zeros(rows * 1024, dtype=torch.uint8),
                    torch.zeros(pages * 1024, dtype=torch.uint8),
                    torch.zeros(reqs * width, dtype=torch.int32),
                    torch.zeros(rows, dtype=torch.int32), torch.zeros(rows, dtype=torch.int32),
                    torch.zeros(reqs, dtype=torch.int32),
                    torch.zeros(rows * columns, dtype=torch.float32),
                    torch.zeros(rows, dtype=torch.int32), torch.zeros(rows, dtype=torch.int32))
            gx, gy = K.wmma_grid(rows, columns, 16, launch.cols_per_block)
            flyc.compile(launch, *args, gx, gy, rows, columns, pages, reqs, width,
                         fx.Stream(None))
            rec.update(compiled=True, kernel_name=launch.kernel_name)
            print(f"[COMPILED] bf16_r16_tdm_nobar kernel={launch.kernel_name}")
        except Exception as exc:  # noqa: BLE001
            rc = 1
            rec.update(compiled=False, error=f"{type(exc).__name__}: {exc}"[:6000])
            print(f"[FAILED]   bf16_r16_tdm_nobar: {rec['error'][:3000]}")
        out.append(rec)
    dest = os.environ.get("QSA_RESULTS", ".")
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "compile_draft.json"), "w") as fh:
        json.dump({"module": how, "variants": out}, fh, indent=2)
    return rc


if __name__ == "__main__":
    sys.exit(main())
