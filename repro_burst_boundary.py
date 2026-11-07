import os, sys
os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")
sys.path.insert(0, "/app/aiter-test/op_tests/multigpu_tests")
import torch
import torch.distributed as dist
import test_mega_moe_v2 as T
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

# Wedge position scales with the epoch-slot count in units of BURSTS, not calls:
#   D=2 -> wedged in burst 3,  D=8 -> wedged in burst 10  (61 calls per burst).
# That points at the burst BOUNDARY leaking one epoch each time, not at the
# burst depth itself. Same total calls, different number of boundaries:
#   P1: 1 boundary   (one 610-call burst, single sync at the end)
#   P2: 10 boundaries (ten 61-call bursts)
# P1 passing while P2 wedges => the boundary leaks the epoch, and the fix
# belongs at whatever the boundary does, not in the buffer depth.

DECODE_BS = [5, 5, 3, 5, 8, 5, 6, 6]
MTPR = 8192
TOTAL = 610


def burst_run(moe, net, rank, device, tag, local_bs, per_burst):
    x, wts, ids = T._make_inputs(
        local_bs, net["model_dim"], net["experts"], net["topk"], rank, 123, device
    )
    done = 0
    while done < TOTAL:
        n = min(per_burst, TOTAL - done)
        for _ in range(n):
            moe.forward(x, wts, ids, config_tokens=MTPR)
        torch.cuda.synchronize()
        done += n
        if rank == 0:
            print(f"{tag} {done}/{TOTAL} (boundaries so far={done // per_burst})", flush=True)
    dist.barrier()
    if rank == 0:
        print(f"BOUNDARY_PASSED {tag} per_burst={per_burst} total={done}", flush=True)


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
        print("BOUNDARY_START", flush=True)
    burst_run(moe, net, rank, device, "P1_one_big_burst", tiny, TOTAL)
    burst_run(moe, net, rank, device, "P2_ten_bursts", tiny, 61)
    if rank == 0:
        print("BOUNDARY_ALL_DONE", flush=True)
    T._cleanup()


if __name__ == "__main__":
    main()
