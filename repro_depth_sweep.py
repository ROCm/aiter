import os, sys
os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")
sys.path.insert(0, "/app/aiter-test/op_tests/multigpu_tests")
import torch
import torch.distributed as dist
import test_mega_moe_v2 as T
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

# Is a deeper epoch buffer an actual FIX, or just a bigger window?
# If D epoch slots only buy tolerance proportional to D, then a launch burst
# deeper than what D covers must still wedge. Sweep the burst depth well past
# ATOM's 61 and see whether a fixed D holds at any depth.
#   D=8 holding at depth 488  -> drift is bounded by data dependencies, small
#                                constant D is a real fix.
#   D=8 wedging at some depth -> D only widens the window; needs a real bound.

DECODE_BS = [5, 5, 3, 5, 8, 5, 6, 6]
MTPR = 8192
SLOTS = os.environ.get("AITER_MEGA_EPOCH_SLOTS", "2")
DEPTHS = [int(d) for d in os.environ.get("SWEEP_DEPTHS", "61,122,244,488").split(",")]
# Hold total work per depth roughly constant so a deep burst is not simply
# given more calls in which to trip.
TOTAL_PER_DEPTH = int(os.environ.get("SWEEP_TOTAL", "2000"))


def phase(moe, net, rank, device, depth):
    # SWEEP_BS=0 -> the sparse DP-attention decode shape (few tokens per rank,
    # so most rank pairs exchange nothing in a given call). SWEEP_BS=N -> a
    # uniform dense batch where every rank sends to every rank.
    bs_env = int(os.environ.get("SWEEP_BS", "0"))
    local_bs = bs_env or DECODE_BS[rank % len(DECODE_BS)]
    x, wts, ids = T._make_inputs(
        local_bs, net["model_dim"], net["experts"], net["topk"], rank, 123, device
    )
    rounds = max(1, TOTAL_PER_DEPTH // depth)
    for r in range(rounds):
        for _ in range(depth):
            moe.forward(x, wts, ids, config_tokens=MTPR)
        torch.cuda.synchronize()
        if rank == 0 and (r + 1) % max(1, rounds // 4) == 0:
            print(f"depth={depth} {(r + 1) * depth}/{rounds * depth} ok", flush=True)
    dist.barrier()
    if rank == 0:
        print(f"SWEEP_PASSED slots={SLOTS} depth={depth} calls={rounds * depth}", flush=True)


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
    if rank == 0:
        print(f"SWEEP_START slots={SLOTS} depths={DEPTHS}", flush=True)
    for d in DEPTHS:
        phase(moe, net, rank, device, d)
    if rank == 0:
        print("SWEEP_ALL_DONE", flush=True)
    T._cleanup()


if __name__ == "__main__":
    main()
