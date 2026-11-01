# SPDX-License-Identifier: MIT
"""No-device stock FlyDSL real-ISA trace of conversion specialization only."""
import math
import os
os.environ.setdefault("COMPILE_ONLY", "1")
import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from _load import load_kernel_module
K, how = load_kernel_module()
rows, columns, pages, reqs, width = 32, 64, 16, 1, 16
args = (torch.zeros(rows * 4 * 128 * 2, dtype=torch.uint8),
        torch.zeros(pages * 4 * 128, dtype=torch.uint8),
        torch.zeros(pages * 4, dtype=torch.float32),
        torch.zeros(reqs * width, dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32), torch.zeros(rows, dtype=torch.int32),
        torch.zeros(reqs, dtype=torch.int32), torch.zeros(rows * columns, dtype=torch.float32),
        torch.zeros(rows, dtype=torch.int32), torch.zeros(rows, dtype=torch.int32))
launcher = K._build_wmma_fp8_kernel(math.sqrt(128), 32, "tdm")
compiled = flyc.compile(launcher, *args, 1, 1, rows, columns, pages, reqs, width, fx.Stream(None))
print("[COMPILED]", launcher.kernel_name, type(compiled).__name__, how)
