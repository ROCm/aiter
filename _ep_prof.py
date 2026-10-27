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
    ts = eng._dbg.view(-1, 19).to(torch.int64).cpu().numpy()
    nw8 = bool((ts[:, 9] > 0).all())
    s9 = ts[:, 9] if nw8 else ts[:, 1]
    # (name, from-slot, to-slot, is_comm); s_memrealtime is 100 MHz -> /1e5 = ms
    PH = [
        ("barrier_init", ts[:, 0], ts[:, 1], 1),
        ("disp_send", ts[:, 1], s9, 1),
        ("disp_meta_wait", s9, ts[:, 4], 1),
        ("plan", ts[:, 4], ts[:, 5], 0),
        ("gather", ts[:, 5], ts[:, 10], 0),
        ("payload_wait", ts[:, 10], ts[:, 11], 1),
        ("gemm1", ts[:, 11], ts[:, 7], 0),
        ("gemm2+comb_send", ts[:, 7], ts[:, 8], 0),
        ("tail", ts[:, 8], ts[:, 2], 0),
        ("comb_wait", ts[:, 2], ts[:, 12], 1),
        ("comb_reduce", ts[:, 12], ts[:, 3], 0),
    ]
    vals = [prep, ker, (ts[:, 3].max() - ts[:, 0].min()) / 1e5]
    for _, a0, a1, _c in PH:
        d = (a1 - a0) / 1e5
        vals += [float(d.mean()), float(d.max())]
    SPL = ["g1 kloop", "g1 pre-epi cbar", "g1 epilogue", "g2 kloop", "g2 pre-epi cbar", "g2 epilogue"]
    for k in range(6):
        d = (ts[:, 13 + k] & 0xFFFFFFFF) / 1e5
        vals += [float(d.mean()), float(d.max())]
    dev = torch.tensor(vals, device=DEV, dtype=torch.float64)
    gath = [torch.empty_like(dev) for _ in range(WORLD)]
    dist.all_gather(gath, dev)
    if RANK == 0:
        g = np.array([v.cpu().numpy() for v in gath]).max(0)
        print(f"[PROF] bs={bs} nw8={nw8} prep={g[0]:.3f} kernel={g[1]:.3f} "
              f"dev_span={g[2]:.3f} ms  (per phase: mean / max over CTAs, max over ranks)",
              flush=True)
        comm = 0.0
        for i, (n, _a0, _a1, c) in enumerate(PH):
            mean, mxv = g[3 + 2 * i], g[4 + 2 * i]
            comm += mean if c else 0.0
            print(f"[PROF]   {'*' if c else ' '} {n:16s} {mean:7.3f} / {mxv:7.3f}", flush=True)
        o = 3 + 2 * len(PH)
        for k, n in enumerate(SPL):
            print(f"[PROF]   # {n:16s} {g[o + 2 * k]:7.3f} / {g[o + 2 * k + 1]:7.3f}", flush=True)
        print(f"[PROF]   comm(*) sum of means = {comm:.3f} ms = "
              f"{100 * comm / g[2]:.1f}% of dev_span", flush=True)
    dist.barrier()
dist.destroy_process_group()
