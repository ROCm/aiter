import os, sys
os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")
sys.path.insert(0, "/app/aiter-test/op_tests/multigpu_tests")
import torch
import torch.distributed as dist
import test_mega_moe_v2 as T
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

# Reproduced: N mega calls issued back-to-back with no host sync between them
# wedges all ranks (GPUs spin at 100%). One call + synchronize never wedges.
# ATOM issues 61 (one per MoE layer) per forward step, which is why serving
# always hung and every prior isolation test passed.
# This ladder finds the smallest N that wedges, and whether asymmetry matters.

DECODE_BS = [5, 5, 3, 5, 8, 5, 6, 6]
MTPR = 8192


def trial(moe, net, rank, device, tag, local_bs, back_to_back, steps=5):
    x, wts, ids = T._make_inputs(
        local_bs, net["model_dim"], net["experts"], net["topk"], rank, 123, device
    )
    for step in range(steps):
        for _ in range(back_to_back):
            moe.forward(x, wts, ids, config_tokens=MTPR)
        torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(f"PASSED {tag} n={back_to_back} bs={local_bs}", flush=True)


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

    if rank == 0:
        print("LADDER_START", flush=True)
    # How many back-to-back calls does it take, with the serving token pattern?
    for n in (1, 2, 3, 4, 8, 16, 32, 61):
        trial(moe, net, rank, device, "asym_tiny", tiny, n)
    # Same depth, but uniform token counts: is asymmetry required?
    for n in (8, 61):
        trial(moe, net, rank, device, "uniform_tiny", 8, n)
    # Same depth, prefill-sized tokens: is it specific to small batches?
    for n in (8, 61):
        trial(moe, net, rank, device, "uniform_large", 2048, n)

    if rank == 0:
        print("LADDER_ALL_DONE", flush=True)
    T._cleanup()


if __name__ == "__main__":
    main()
