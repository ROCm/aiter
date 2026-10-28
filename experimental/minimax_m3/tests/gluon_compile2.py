"""Compile-only build of the Gluon M3 step kernel (one GPU context, nothing launched): catches
source / layout errors before a run takes GPU queue time, and prints resources.

    HIP_VISIBLE_DEVICES=3 python3 tests/gluon_compile2.py [S ...]   (default S=4, QL=4; env QL, STAGES,
    VLLM_M3_GLUON_SRC honoured through the runner's package redirect)"""
import os
import sys
import time

import torch

from vllm.models.minimax_m3.amd.mono.gluon import runner as R

LY = R.LY
m3_step_kernel = R._kernel()
dev = torch.device("cuda")
QL = int(os.environ.get("QL", "4"))
STAGES = int(os.environ.get("STAGES", "511"))
for n in [int(a) for a in sys.argv[1:]] or [4]:
    t0 = time.time()
    z64 = torch.zeros(64, dtype=torch.int64, device=dev)
    i32 = torch.zeros(64, 4, dtype=torch.int32, device=dev)
    h = torch.zeros(64, LY.HIDDEN, dtype=torch.bfloat16, device=dev)
    u8 = torch.zeros(1 << 20, dtype=torch.uint8, device=dev)
    w5 = torch.zeros(64, 5, dtype=torch.float32, device=dev)
    pr = LY.peer_regions()
    if getattr(LY, "KARG_V3", False):
        args = (z64, 57, 3, z64, u8, u8, w5, h, z64, z64, i32, 4, i32, h, h, h, h, u8, z64, i32, z64, 0, 0.088,
                1e-6, 2.0, 1.0, 7.0, h, i32, w5, h, h, h)
    elif getattr(LY, "KARG_V2", False):
        args = (z64, 57, 3, z64, u8, i32, z64, 0, h, z64, z64, i32, 4, i32, h, h, h, h, u8, z64, 0.088, 1e-6, 2.0,
                1.0, 7.0, h, i32, w5, h, h, h)
    else:
        args = (z64, 57, h, z64, z64, i32, 4, i32, h, h, h, h, u8, u8, z64, z64, i32, z64, 0, 0.088, 1e-6, 2.0, 1.0,
                7.0, 3, h, i32, w5, h, h, h)
    cc = m3_step_kernel.warmup(*args, grid=(LY.BLOCKS,), S=n, HALF=LY.StepLayout().half(), PHALF=pr["_half"],
                               FFN_OFF=pr["ffn"], LOOP=bool(int(os.environ.get("LOOP", "0"))), STAGES=STAGES, QL=QL, NL=int(os.environ.get('NL', '1')), num_warps=8,
                               enable_fp_fusion=False)
    cc._init_handles()
    asm = cc.asm["amdgcn"]
    print(f"S={n} QL={QL}: {time.time() - t0:.0f}s VGPRs {cc.n_regs} spills {cc.n_spills} LDS {cc.metadata.shared} "
          f"text {len(asm)} B, s_and_saveexec {asm.count('s_and_saveexec')} scratch_ {asm.count('scratch_')}",
          flush=True)
