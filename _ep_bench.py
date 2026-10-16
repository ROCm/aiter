"""8-GPU a4w4 v4_pro perf comparison: one-launch EP (new) vs two-stage MegaMoEV2 vs
tensor-parallel MegaMoeTP. All measured identically -- eager forward() wall-clock
(warmup + N iters, dist-synced) -- so it is apples-to-apples (not graph-replay).
Perf only: each engine gets valid (independently generated) a4w4 weights of its own
sharding; outputs are not cross-checked here (correctness is covered by the relL2 tests).
Launch: torchrun --standalone --nproc_per_node=8 _ep_bench.py
"""

import os
import time

import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight, shuffle_weight_a16w4
from aiter.utility import fp4_utils

RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
LR = int(os.environ.get("LOCAL_RANK", RANK))
torch.cuda.set_device(LR)
DEV = torch.device("cuda", LR)
dist.init_process_group("cpu:gloo,cuda:nccl", device_id=DEV)

H, I, K, SWIGLU = 7168, 3072, 6, 10.0
EPR = 48  # per-rank experts fixed across world sizes (8-GPU v4_pro value)
E = EPR * WORLD  # 384 @ world=8, 192 @ world=4 -- keeps per-rank geometry identical
MTPR = 1024
ITERS, WARMUP = 20, 5
qz = aiter.get_torch_quant(aiter.QuantType.per_1x32)


def _q(experts, rows, cols, mag, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(experts, rows, cols, dtype=torch.bfloat16, device=DEV, generator=g) * mag
    q, s = qz(w, quant_dtype=dtypes.fp4x2)
    return q.view(experts, rows, cols // 2).contiguous(), s, s.view(experts, rows, cols // 32).contiguous()


def build_ep():
    from aiter.ops.flydsl.kernels.mega_moe_ep.mega_moe_ep import MegaMoeEPEngine

    w1q, _, w1s = _q(EPR, 2 * I, H, H**-0.25, 100 + RANK)
    w2q, _, w2s = _q(EPR, H, I, I**-0.25, 200 + RANK)
    return MegaMoeEPEngine(
        rank=RANK, world_size=WORLD, model_dim=H, inter_dim=I, experts=E, topk=K,
        max_tok_per_rank=MTPR, w1=w1q, w1_scale=w1s, w2=w2q, w2_scale=w2s,
        quant="a4w4", act="silu", swiglu_limit=SWIGLU, device=DEV,
    )


def build_v2():
    from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

    w1q, w1sr, _ = _q(EPR, 2 * I, H, H**-0.25, 100 + RANK)
    w2q, w2sr, _ = _q(EPR, H, I, I**-0.25, 200 + RANK)
    return MegaMoEV2(
        rank=RANK, world_size=WORLD, model_dim=H, inter_dim=I, experts=E, topk=K,
        quant="a4w4",
        w1=shuffle_weight_a16w4(w1q, 16, True).contiguous(),
        w1_scale=shuffle_scale_a16w4(w1sr, EPR, True).contiguous(),
        w2=shuffle_weight_a16w4(w2q, 16, False).contiguous(),
        w2_scale=shuffle_scale_a16w4(w2sr, EPR, False).contiguous(),
        max_tok_per_rank=MTPR, swiglu_limit=SWIGLU,
    )


def build_tp():
    from aiter.ops.flydsl.kernels.mega_moe_tp.mega_moe_tp import MegaMoeTPEngine

    ish = I // WORLD  # TP shards the inter dim; every rank holds all E experts
    w1q, w1sr, _ = _q(E, 2 * ish, H, H**-0.25, 300 + RANK)
    w2q, w2sr, _ = _q(E, H, ish, ish**-0.25, 400 + RANK)
    return MegaMoeTPEngine(
        rank=RANK, world_size=WORLD, model_dim=H, inter_dim=ish, experts=E, topk=K,
        max_local_tokens=MTPR,
        w1=shuffle_weight(w1q, layout=(16, 16)).contiguous(),
        w1_scale=fp4_utils.e8m0_shuffle(w1sr).contiguous(),
        w2=shuffle_weight(w2q, layout=(16, 16)).contiguous(),
        w2_scale=fp4_utils.e8m0_shuffle(w2sr).contiguous(),
        activation="silu", act_dtype="fp4", swiglu_limit=SWIGLU, device=DEV,
    )


def make_inputs(bs):
    g = torch.Generator(device=DEV).manual_seed(777 + RANK * 131 + bs)
    x = torch.randn(bs, H, dtype=torch.bfloat16, device=DEV, generator=g) * 0.1
    wts, ids = torch.topk(torch.softmax(torch.randn(bs, E, device=DEV, generator=g), -1), K, -1)
    return x, wts.to(torch.float32).contiguous(), ids.to(torch.int32).contiguous()


def bench(engine, bs):
    x, wts, ids = make_inputs(bs)
    for _ in range(WARMUP):
        engine(x, wts, ids)
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        engine(x, wts, ids)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / ITERS * 1e3
    out = torch.tensor([ms], device=DEV)
    gathered = [torch.empty_like(out) for _ in range(WORLD)]
    dist.all_gather(gathered, out)
    return max(x.item() for x in gathered)  # worst rank = effective e2e


# One engine per process (env ENGINE): the three use incompatible IPC/shmem backends
# (EP/TP hipIpc SymmetricArena, V2 mori shmem), so benching them in separate torchrun
# runs avoids cross-backend state corruption.
ENGINE = os.environ["ENGINE"]
if ENGINE == "V2":
    import torch._C._distributed_c10d as c10d

    import mori.shmem as ms

    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")

engine = {"EP": build_ep, "V2": build_v2, "TP": build_tp}[ENGINE]()
if RANK == 0:
    print(f"built {ENGINE}", flush=True)
dist.barrier()

for bs in (1, 2, 4, 8, 64, 256, 512, 1024):
    try:
        e2e = bench(engine, bs)
        if RANK == 0:
            print(f"[{ENGINE}] bs={bs:<5d} e2e={e2e:.3f}ms", flush=True)
    except Exception as exc:  # noqa: BLE001
        if RANK == 0:
            print(f"[{ENGINE}] bs={bs} RUN FAIL: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    dist.barrier()

if ENGINE == "V2":
    ms.shmem_finalize()
dist.barrier()
dist.destroy_process_group()
