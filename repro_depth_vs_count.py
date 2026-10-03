import os, sys
os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")
sys.path.insert(0, "/app/aiter-test/op_tests/multigpu_tests")
import torch
import torch.distributed as dist
import test_mega_moe_v2 as T
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

# Does the wedge depend on how many mega calls are issued BACK TO BACK (launch
# pipeline depth), or merely on how MANY calls have been made in total (an
# accumulating race)? The earlier ladder could not tell these apart: it ran the
# depths cumulatively in one process, so "wedged at depth 32" and "wedged after
# ~200 total calls" are the same observation.
#
# A: depth 1  (host sync after every single call) for TOTAL calls
# B: depth 61 (ATOM's per-forward-step burst)     for TOTAL calls
# A survives + B wedges  -> depth is the trigger.
# A wedges               -> depth is irrelevant, it is a cumulative race.

DECODE_BS = [5, 5, 3, 5, 8, 5, 6, 6]
MTPR = 8192
TOTAL = 610


def phase(moe, net, rank, device, tag, local_bs, depth):
    x, wts, ids = T._make_inputs(
        local_bs, net["model_dim"], net["experts"], net["topk"], rank, 123, device
    )
    done = 0
    while done < TOTAL:
        for _ in range(depth):
            moe.forward(x, wts, ids, config_tokens=MTPR)
        torch.cuda.synchronize()
        done += depth
        if rank == 0 and done % 100 < depth:
            print(f"{tag} progress {done}/{TOTAL}", flush=True)
    dist.barrier()
    if rank == 0:
        print(f"PHASE_PASSED {tag} depth={depth} total={done}", flush=True)


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
        print("DVC_START", flush=True)
    phase(moe, net, rank, device, "A_depth1", tiny, 1)
    phase(moe, net, rank, device, "B_depth61", tiny, 61)
    if rank == 0:
        print("DVC_ALL_DONE", flush=True)
    T._cleanup()


if __name__ == "__main__":
    main()
