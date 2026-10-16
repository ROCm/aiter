"""8-GPU EP MegaMoE bring-up: real cross-rank dispatch/combine via the hipIpc arena +
the one-launch kernel, compared to the distributed ep_reference oracle (all-gather MoE).

Each rank owns E//world local experts (natural MXFP4 weights, fed to BOTH the device
engine and the oracle). Tokens route to global expert ids 0..E, so dispatch_send does a
genuine all-to-all P2P scatter to the owner ranks' arenas and combine_send scatters the
GEMM2 partials back -- exercising the cross-rank arrival counters + the init barrier
(the C/D logic that single-rank could not stress). Launch: torchrun --nproc_per_node=8.
"""

import os

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

H, I, E, K = 7168, 3072, 384, 6
EPR = E // WORLD
MTPR = 1024
QUANT = os.environ.get("EP_QUANT", "a8w4")
TOL = 0.25 if QUANT == "a4w4" else 0.10

# this rank's local experts (natural / unshuffled MXFP4), distinct per rank
qz = aiter.get_torch_quant(aiter.QuantType.per_1x32)
gw = torch.Generator(device=DEV).manual_seed(1234 + RANK)
w1 = torch.randn(EPR, 2 * I, H, dtype=torch.bfloat16, device=DEV, generator=gw) * (H**-0.25)
w1_q, w1_s = qz(w1, quant_dtype=dtypes.fp4x2)
w1_q = w1_q.view(EPR, 2 * I, H // 2).contiguous()
w1_s = w1_s.view(EPR, 2 * I, H // 32).contiguous()
w2 = torch.randn(EPR, H, I, dtype=torch.bfloat16, device=DEV, generator=gw) * (I**-0.25)
w2_q, w2_s = qz(w2, quant_dtype=dtypes.fp4x2)
w2_q = w2_q.view(EPR, H, I // 2).contiguous()
w2_s = w2_s.view(EPR, H, I // 32).contiguous()
del w1, w2
torch.cuda.empty_cache()

engine = MegaMoeEPEngine(
    rank=RANK, world_size=WORLD, model_dim=H, inter_dim=I, experts=E, topk=K,
    max_tok_per_rank=MTPR, w1=w1_q, w1_scale=w1_s, w2=w2_q, w2_scale=w2_s,
    quant=QUANT, act="silu", swiglu_limit=10.0, device=DEV,
)
if RANK == 0:
    print(f"8-GPU EP: n_cta={engine.n_cta} epr={EPR} quant={QUANT} tol={TOL}", flush=True)


def run(bs):
    dist.barrier()  # keep all ranks on the same bs / launch epoch
    g = torch.Generator(device=DEV).manual_seed(777 + RANK * 131 + bs)
    x = torch.randn(bs, H, dtype=torch.bfloat16, device=DEV, generator=g) * 0.1
    logits = torch.randn(bs, E, device=DEV, generator=g)
    wts, ids = torch.topk(torch.softmax(logits, dim=-1), K, dim=-1)
    ids = ids.to(torch.int32).contiguous()
    wts = wts.to(torch.float32).contiguous()
    engine._part.local.zero_()
    out = engine.forward(x, wts, ids)
    torch.cuda.synchronize()
    err = engine.poll_errors()
    recv = engine._recv.local
    nrecv = recv.ge(0).logical_and(recv.lt(EPR)).sum().item()
    partv = engine._part.local.view(MTPR, K, H)[:bs]
    routes_nz = partv.float().abs().sum(-1).gt(1e-6)  # [bs, K]
    complete = routes_nz.all(-1).sum().item()
    print(f"[r{RANK}] bs={bs} err={err} recv_valid={nrecv} complete_tok={complete}/{bs}", flush=True)
    ref = engine.reference_forward(x, wts, ids, (w1_q, w1_s, w2_q, w2_s))
    o, r = out.float(), ref.float()
    rel = ((o - r).norm() / r.norm().clamp_min(1e-9)).item()
    stats = torch.tensor([rel, float(err)], device=DEV)
    gathered = [torch.empty_like(stats) for _ in range(WORLD)]
    dist.all_gather(gathered, stats)
    if RANK == 0:
        worst = max(x[0].item() for x in gathered)
        anyerr = any(x[1].item() for x in gathered)
        tag = "PASS" if (worst < TOL and not anyerr) else "FAIL"
        per = " ".join(f"r{i}={x[0].item():.3f}/e{int(x[1].item())}" for i, x in enumerate(gathered))
        print(f"  bs={bs:<5d} worst_relL2={worst:.4f} err={anyerr} {tag}  [{per}]", flush=True)


for bs in (8, 64, 512):
    run(bs)
dist.barrier()
dist.destroy_process_group()
