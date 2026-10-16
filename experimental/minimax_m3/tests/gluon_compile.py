"""Compile-only build of the Gluon M3 step kernel at several widths (one GPU, no launch):
resources and ISA checks. Usage: python3 tests/gluon_compile.py [S ...]"""
import sys
import time

import torch

from vllm.models.minimax_m3.amd.mono.gluon import layout as LY
from vllm.models.minimax_m3.amd.mono.gluon.kernel import m3_step_kernel

dev = torch.device("cuda")
import os
STAGES = int(os.environ.get('STAGES', '511'))
widths = [int(a) for a in sys.argv[1:]] or [1]
for n in widths:
    t0 = time.time()
    z64 = torch.zeros(16, dtype=torch.int64, device=dev)
    i32 = torch.zeros(16, 4, dtype=torch.int32, device=dev)
    h = torch.zeros(16, LY.HIDDEN, dtype=torch.bfloat16, device=dev)
    u8 = torch.zeros(1024, dtype=torch.uint8, device=dev)
    pr = LY.peer_regions()
    args = (z64, 57, h, z64, z64, i32, 4, i32, h, h, h, h, u8, 0, z64, i32, i32, z64, 0, 0.088, 1e-6, 2.0, 1.0, 7.0, 3)
    cc = m3_step_kernel.warmup(*args, grid=(256,), S=n, HALF=LY.StepLayout().half(), PHALF=pr["_half"],
                               FFN_OFF=pr["ffn"], num_warps=8, enable_fp_fusion=False, STAGES=STAGES, LOOP=bool(int(os.environ.get('LOOP', '0'))))
    cc._init_handles()
    asm = cc.asm["amdgcn"]
    print(f"S={n}: {time.time()-t0:.0f}s VGPRs {cc.n_regs} spills {cc.n_spills} LDS {cc.metadata.shared} "
          f"waterfall {asm.count('s_and_saveexec')} buffer_load_dwordx4 {asm.count('buffer_load_dwordx4')} "
          f"ds_read_u8 {asm.count('ds_read_u8')} scratch {asm.count('scratch_')}", flush=True)
    open(f"/home/vigokhal/m3_monokernel/tests/probes/gluon_s{n}_st{STAGES}.s", "w").write(asm)
    open(f"/home/vigokhal/m3_monokernel/tests/probes/gluon_s{n}_st{STAGES}.ttgir", "w").write(cc.asm["ttgir"])
