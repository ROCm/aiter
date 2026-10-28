"""Time flydsl compute / asm / tdm a4w4 GEMMs on identical inputs, in a given order.

a4w4_cmp.py M N K [--order tdm,compute,asm] [--idle S] [--init random|const|zero]
                  [--warmup 3] [--iters 64]
"""
import argparse
import os
import time

# The TDM kernel reads its schedule knobs from the environment at import time;
# default them to reproduce.sh's values so the tuned gemm1 kernel is compiled.
for _k, _v in {
    "ROCM_PATH": "/opt/rocm",
    "AITER_FLYDSL_TDM_EARLY_TIMEOUT": "0",
    "AITER_FLYDSL_DISABLE_CLUSTER_SYNC": "1",
    "AITER_FLYDSL_MMA_FIRST_GROUP": "4",
    "AITER_FLYDSL_MMA_GROUP": "4",
    "AITER_FLYDSL_DS_FIRST_N": "8",
    "AITER_FLYDSL_EXPLICIT_VGPR_PARTITION": "1",
    "AITER_FLYDSL_LDS_WAR_ORDER": "1",
    "AITER_FLYDSL_LDS_WAR_ORDER": "1",
    "AITER_FLYDSL_MXFP4_CLUSTER_N": "4",
    "AITER_FLYDSL_MXFP4_CLUSTER_M": "2",
}.items():
    os.environ.setdefault(_k, _v)

import torch  # noqa: E402

import aiter
from aiter import dtypes
from aiter.ops.flydsl.a4w4_mxscale_bpreshuffle_gemm_gfx1250 import (
    run_gemm_a4w4_mxscale_bpreshuffle_gfx1250,
)
from aiter.ops.flydsl.a4w4_tdm_gemm_gfx1250 import (
    gemm_a4w4_tdm,
    prepare_a4w4_tdm_a,
    prepare_a4w4_tdm_b_scale,
)
from aiter.ops.gemm_op_a4w4 import gemm_mxfp4_asm
from aiter.ops.shuffle import shuffle_mxfp8fp4_scale, shuffle_weight
from aiter.test_common import checkAllclose, run_perftest

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("M", type=int)
parser.add_argument("N", type=int)
parser.add_argument("K", type=int)
parser.add_argument("--order", default="asm,tdm,compute", help="comma list of compute,asm,tdm")
parser.add_argument("--idle", type=float, default=2.0, help="seconds idle before each kernel")
parser.add_argument(
    "--init",
    choices=("random", "const", "zero"),
    default="random",
    help="x/w init: randn, ones, or zeros",
)
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--iters", type=int, default=64)
args = parser.parse_args()
M, N, K = args.M, args.N, args.K
order = args.order.split(",")
COMPUTE = "flydsl_a4w4_bpreshuffle_compute_wmma_t256x256x256_mw2_nw2_nb4_sk1_cm4_cn4"
INIT = {"random": torch.randn, "const": torch.ones, "zero": torch.zeros}[args.init]

torch.set_default_device("cuda")
torch.manual_seed(0)
dtype = torch.bfloat16
quant = aiter.get_hip_quant(aiter.QuantType.per_1x32)
x = INIT((M, K), dtype=dtype)
w = INIT((N, K), dtype=dtype)
wq_raw, ws_raw = quant(w, quant_dtype=dtypes.fp4x2, shuffle=False)
wq = shuffle_weight(wq_raw, layout=(16, 16))
ws = shuffle_mxfp8fp4_scale(ws_raw)
xq, xs = quant(x, quant_dtype=dtypes.fp4x2, scale_layout_m32k4=True)
xq_t, xs_t = prepare_a4w4_tdm_a(x)
ws_t = prepare_a4w4_tdm_b_scale(ws_raw.view(torch.uint8)[:N])
out_c = torch.empty((M, N), dtype=dtype)

fns = {
    "compute": lambda: run_gemm_a4w4_mxscale_bpreshuffle_gfx1250(
        xq.view(torch.uint8), wq.view(torch.uint8), xs, ws, out_c, COMPUTE
    ),
    "asm": lambda: gemm_mxfp4_asm(xq, wq, xs, ws, dtype=dtype, a_preshuffle=True),
    "tdm": lambda: gemm_a4w4_tdm(xq_t, xs_t, wq, ws_t, dtype=dtype),
}
ref = fns["asm"]().clone()
for name in order:
    fns[name]()
torch.cuda.synchronize()

res = []
for name in order:
    time.sleep(args.idle)
    out, us = run_perftest(fns[name], num_warmup=args.warmup, num_iters=args.iters)
    checkAllclose(ref, out, msg=f"{name} vs asm")
    res.append(f"{name}={us:.1f}")
print(
    f"RESULT init={args.init} warmup={args.warmup} iters={args.iters} "
    f"order={','.join(order)} " + " ".join(res),
    flush=True,
)
