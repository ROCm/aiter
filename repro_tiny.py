import os, sys
os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")
sys.path.insert(0, "/app/aiter-test/op_tests/multigpu_tests")
import torch
import test_mega_moe_v2 as T
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

# Observed at the ATOM serving wedge: all 8 ranks inside the SAME mega call
# (calls=5551 everywhere) with tiny, asymmetric decode token counts while the
# config was pinned to 8192. Never covered by prior isolation tests (min bs=64).
DECODE_BS = [5, 5, 3, 5, 8, 5, 6, 6]

MTPR = 8192


def run(moe, net, rank, device, tag, local_bs, cfg, iters=3):
    x, wts, ids = T._make_inputs(
        local_bs, net["model_dim"], net["experts"], net["topk"], rank, 123, device
    )
    for it in range(iters):
        print(f"[{tag}] rank={rank} it={it} bs={local_bs} cfg={cfg} ENTER", flush=True)
        out = moe.forward(x, wts, ids, config_tokens=cfg)
        torch.cuda.synchronize()
        print(f"[{tag}] rank={rank} it={it} DONE out={tuple(out.shape)}", flush=True)


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

    # S1: uniform tiny, config pinned to mtpr (isolates "tiny tokens + big config")
    run(moe, net, rank, device, "S1_uniform_tiny_cfg8192", 8, MTPR)
    # S2: asymmetric tiny (exact serving pattern), config pinned
    run(moe, net, rank, device, "S2_asym_tiny_cfg8192", tiny, MTPR)
    # S3: asymmetric tiny, natural per-rank config (what upstream does by default)
    run(moe, net, rank, device, "S3_asym_tiny_cfgnone", tiny, None)
    # S4: prefill/decode alternation on the shared instance (serving-like)
    for it in range(3):
        run(moe, net, rank, device, f"S4_prefill_r{it}", 1024 + rank * 8, MTPR, iters=1)
        run(moe, net, rank, device, f"S4_decode_r{it}", tiny, MTPR, iters=1)

    T._barrier()
    if rank == 0:
        print("[TINY] ALL_DONE", flush=True)
    T._cleanup()


if __name__ == "__main__":
    main()
