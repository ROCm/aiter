"""Compile-only check of the MegaMoE TP gfx1250 kernel (no launch, one process)."""
import os
import sys
import time

import flydsl.expr as fx
import torch

from aiter.ops.flydsl.kernels.mega_moe_tp_1250.engine import default_launch_cfg
from aiter.ops.flydsl.kernels.mega_moe_tp_1250.kernel import compile_mega_moe_tp_1250
from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled, ptr_arg

tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 64
tp = int(os.environ.get("TP", "4"))
cfg = default_launch_cfg(tokens, int(os.environ.get("HID", "4096")))
t0 = time.time()
fn, ncta = compile_mega_moe_tp_1250(H=int(os.environ.get("HID", "4096")), I=(int(os.environ.get("INTER_PAD", 0)) or int(os.environ.get("INTER_TOTAL", "2560")) // tp + 127) // 128 * 128, Ip=int(os.environ.get("INTER_PAD", 0)) or int(os.environ.get("INTER_TOTAL", "2560")) // tp, E=int(os.environ.get("EXP", "256")), TOPK=int(os.environ.get("TOPK", "6")), tp=tp,
                                    mmax=max(tokens // tp, 1), act=os.environ.get("ACT", "swiglu"), n_cu=256, cfg=cfg)
import flydsl.compiler as flyc
P = flyc.from_c_void_p(fx.Int32, 4096)
args = [P] * 10 + [4096] * 19 + [0] * 7 + [0, 1, 64, 7.0, 1.0, 1.0] + [P] * 8 + [4096] * 2 + [0] * 2 + [fx.Stream(None)]
_preload_compiled(fn, *args)
print(f"compiled {cfg} ncta={ncta} in {time.time() - t0:.1f}s")
