"""8-GPU host-phase decomposition of the EP forward: prep (quant+alloc+recv-fill)
vs kernel (launch->sync). Diagnostic only. torchrun --standalone --nproc_per_node=8.
"""

import os

import numpy as np
import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.ops.flydsl.kernels.mega_moe_ep.mega_moe_ep import MegaMoeEPEngine

RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
LR = int(os.environ.get("LOCAL_RANK", RANK))
torch.cuda.set_device(LR)
DEV = torch.device("cuda", LR)
dist.init_process_group("cpu:gloo,cuda:nccl", device_id=DEV)

H, I, K = 7168, 3072, 6
EPR = 48
E = EPR * WORLD
MTPR = int(os.environ.get("EP_MTPR", "1024"))
qz = aiter.get_torch_quant(aiter.QuantType.per_1x32)
gw = torch.Generator(device=DEV).manual_seed(1234 + RANK)
w1 = torch.randn(EPR, 2 * I, H, dtype=torch.bfloat16, device=DEV, generator=gw) * (H**-0.25)
w1q, w1s = qz(w1, quant_dtype=dtypes.fp4x2)
w1q = w1q.view(EPR, 2 * I, H // 2).contiguous()
w1s = w1s.view(EPR, 2 * I, H // 32).contiguous()
w2 = torch.randn(EPR, H, I, dtype=torch.bfloat16, device=DEV, generator=gw) * (I**-0.25)
w2q, w2s = qz(w2, quant_dtype=dtypes.fp4x2)
w2q = w2q.view(EPR, H, I // 2).contiguous()
w2s = w2s.view(EPR, H, I // 32).contiguous()
del w1, w2
eng = MegaMoeEPEngine(
    rank=RANK, world_size=WORLD, model_dim=H, inter_dim=I, experts=E, topk=K,
    max_tok_per_rank=MTPR, w1=w1q, w1_scale=w1s, w2=w2q, w2_scale=w2s,
    quant="a4w4", act="silu", swiglu_limit=10.0, device=DEV,
)


def inp(bs):
    g = torch.Generator(device=DEV).manual_seed(777 + RANK * 131 + bs)
    x = torch.randn(bs, H, dtype=torch.bfloat16, device=DEV, generator=g) * 0.1
    wts, ids = torch.topk(torch.softmax(torch.randn(bs, E, device=DEV, generator=g), -1), K, -1)
    return x, wts.float().contiguous(), ids.int().contiguous()


for bs in [int(b) for b in os.environ.get("EP_BS", "8,512").split(",")]:
    x, wts, ids = inp(bs)
    for _ in range(5):
        eng.forward(x, wts, ids)
    torch.cuda.synchronize()
    dist.barrier()
    eng._prof = []
    for _ in range(30):
        eng.forward(x, wts, ids)
    pr = np.array(eng._prof)
    eng._prof = None
    prep = pr[:, 0].mean() * 1e3
    ker = pr[:, 1].mean() * 1e3
    # device phase split from the last forward's per-CTA timestamps (s_memrealtime,
    # 100 MHz -> ticks/1e5 = ms). phases: barrier_init, roles(dispatch+compute+gemm),
    # combine_final. report worst-CTA (max) per phase.
    # slots: 0 entry, 1 post-barrier_init, 2 post-roles, 3 post-combine_final,
    #        4 post-dispatch_wait, 5 post-dyn_plan
    ts = eng._dbg.view(-1, 9).to(torch.int64).cpu().numpy()
    # for the single active unit per CTA, isolate gemm1/gemm2/combine_send using the
    # per-unit stamps (slots 7,8); only CTAs that ran a unit (slot7>slot5) count.
    act = ts[:, 7] > ts[:, 5]
    bi = (ts[:, 1] - ts[:, 0]) / 1e5  # barrier_init (cross-rank init)
    dw = (ts[:, 4] - ts[:, 1]) / 1e5  # dispatch_wait (compute idle until a2a done)
    dp = (ts[:, 5] - ts[:, 4]) / 1e5  # dyn_plan
    g1 = ((ts[:, 7] - ts[:, 5]) / 1e5)[act]  # gather + GEMM1
    g2 = ((ts[:, 8] - ts[:, 7]) / 1e5)[act]  # GEMM2
    cs = ((ts[:, 2] - ts[:, 8]) / 1e5)[act]  # combine_send + loop tail
    cf = (ts[:, 3] - ts[:, 2]) / 1e5  # combine_final
    tot = (ts[:, 3] - ts[:, 0]) / 1e5

    def mx(a):
        return float(a.max()) if a.size else 0.0

    dev = torch.tensor(
        [prep, ker, bi.max(), dw.max(), dp.max(), mx(g1), mx(g2), mx(cs),
         cf.max(), tot.max()], device=DEV)
    gath = [torch.empty_like(dev) for _ in range(WORLD)]
    dist.all_gather(gath, dev)
    if RANK == 0:
        g = np.array([v.cpu().numpy() for v in gath])
        print(
            f"[PROF] bs={bs:<5d} prep={g[:, 0].max():.3f} kernel={g[:, 1].max():.3f}ms || "
            f"barrier_init={g[:, 2].max():.3f} dispatch_wait={g[:, 3].max():.3f} "
            f"dyn_plan={g[:, 4].max():.3f} | gemm1={g[:, 5].max():.3f} gemm2={g[:, 6].max():.3f} "
            f"combine_send={g[:, 7].max():.3f} combine_final={g[:, 8].max():.3f} "
            f"|| dev_total={g[:, 9].max():.3f}",
            flush=True,
        )
    dist.barrier()
dist.destroy_process_group()
