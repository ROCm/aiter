import os, sys
os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")
sys.path.insert(0, "/app/aiter-test/op_tests/multigpu_tests")
import torch
import torch.distributed as dist
import test_mega_moe_v2 as T
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

# Hypothesis: MegaMoEV2 spin-waits across ranks with a grid sized to the full CU
# count. Under ATOM serving, other kernels (attention, RCCL collectives, async
# output copies) run concurrently on other streams and steal CUs, so some mega
# blocks never become resident and the resident ones spin forever.
# Isolation always passes because nothing else touches the GPU.

DECODE_BS = [5, 5, 3, 5, 8, 5, 6, 6]
MTPR = 8192
LAYERS = 61  # ATOM calls mega once per MoE layer per forward step


def main():
    rank, world, device = T._setup_dist()
    net = T.NETWORKS["v4_pro"]
    le = net["experts"] // world
    packed = T._quantize_weights(net["model_dim"], net["inter_dim"], le, rank, 123, device)
    moe = MegaMoEV2(
        rank=rank, world_size=world, quant="a8w4",
        w1=packed[0], w1_scale=packed[1], w2=packed[2], w2_scale=packed[3],
        max_tok_per_rank=MTPR, **net,
    )

    tiny = DECODE_BS[rank % len(DECODE_BS)]
    x, wts, ids = T._make_inputs(
        tiny, net["model_dim"], net["experts"], net["topk"], rank, 123, device
    )

    side = torch.cuda.Stream(device=device, priority=-1)  # high priority CU hog
    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=device)
    b = torch.randn(8192, 8192, dtype=torch.bfloat16, device=device)
    ar = torch.randn(64 * 1024 * 1024 // 2, dtype=torch.bfloat16, device=device)

    def flood(kind, n):
        with torch.cuda.stream(side):
            for _ in range(n):
                if kind == "gemm":
                    torch.mm(a, b)
                else:
                    dist.all_reduce(ar)

    # C0: control — no contention, same cadence as serving (61 calls per step)
    for step in range(3):
        print(f"[C0_nocontend] rank={rank} step={step} ENTER", flush=True)
        for _ in range(LAYERS):
            moe.forward(x, wts, ids, config_tokens=MTPR)
        torch.cuda.synchronize()
        print(f"[C0_nocontend] rank={rank} step={step} DONE", flush=True)

    # C1: GEMM flood on a high-priority side stream steals CUs during mega
    for step in range(5):
        print(f"[C1_gemm_contend] rank={rank} step={step} ENTER", flush=True)
        flood("gemm", 40)
        for _ in range(LAYERS):
            moe.forward(x, wts, ids, config_tokens=MTPR)
        torch.cuda.synchronize()
        print(f"[C1_gemm_contend] rank={rank} step={step} DONE", flush=True)

    # C2: concurrent RCCL collectives (themselves spin-waiting) during mega
    for step in range(5):
        print(f"[C2_rccl_contend] rank={rank} step={step} ENTER", flush=True)
        flood("ar", 8)
        for _ in range(LAYERS):
            moe.forward(x, wts, ids, config_tokens=MTPR)
        torch.cuda.synchronize()
        print(f"[C2_rccl_contend] rank={rank} step={step} DONE", flush=True)

    # C3: both, with rank-dependent flood depth (staggered arrival like serving)
    for step in range(5):
        print(f"[C3_mixed_contend] rank={rank} step={step} ENTER", flush=True)
        flood("gemm", 10 + rank * 6)
        flood("ar", 2 + rank)
        for _ in range(LAYERS):
            moe.forward(x, wts, ids, config_tokens=MTPR)
        torch.cuda.synchronize()
        print(f"[C3_mixed_contend] rank={rank} step={step} DONE", flush=True)

    T._barrier()
    if rank == 0:
        print("[CONC] ALL_DONE", flush=True)
    T._cleanup()


if __name__ == "__main__":
    main()
