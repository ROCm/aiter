"""MegaMoEV2 deadlock: isolate per-rank ASYMMETRY from CONFIG/token MISMATCH.

Wedging runs so far all had BOTH asymmetric per-rank token counts AND
config_tokens pinned to 8192 while only 3-8 tokens were processed. The two
controls that passed each changed both variables at once, so neither factor is
established. This runs the 2x2.

  BS=0      -> asymmetric per-rank [5,5,3,5,8,5,6,6]
  BS=N      -> uniform N on every rank
  CFG=0     -> stock upstream: forward(x, wts, ids), config from real tokens
  CFG=N     -> forward(..., config_tokens=N)   [needs the local config_tokens patch]

    BS=0 CFG=0     asymmetric, natural config
    BS=8 CFG=8192  uniform,    mismatched config
    BS=0 CFG=8192  known to wedge
    BS=8 CFG=0     known clean
"""

import os, sys

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import test_mega_moe_v2 as T
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

DECODE_BS = [5, 5, 3, 5, 8, 5, 6, 6]
MTPR = 8192
BS = int(os.environ.get("BS", "0"))
CFG = int(os.environ.get("CFG", "0"))
DEPTH = int(os.environ.get("DEPTH", "61"))
TOTAL = int(os.environ.get("TOTAL", "1220"))
VARY = int(os.environ.get("VARY", "1"))  # 1 = reuse one input set (all repros so far)


def main():
    rank, world, device = T._setup_dist()
    net = T.NETWORKS["v4_pro"]
    le = net["experts"] // world
    w = T._quantize_weights(net["model_dim"], net["inter_dim"], le, rank, 123, device)
    moe = MegaMoEV2(
        rank=rank, world_size=world, quant="a8w4",
        w1=w[0], w1_scale=w[1], w2=w[2], w2_scale=w[3],
        max_tok_per_rank=MTPR, **net,
    )
    # BS=-1: the ATOM mixed step under the proposed max_tokens_across_dp fix —
    # one rank prefills 8192 while the rest decode, so the DP-wide max (and thus
    # the shared config) is 8192 and every decode rank is still mismatched.
    if BS == -1:
        bs = MTPR if rank == 0 else DECODE_BS[rank % len(DECODE_BS)]
    else:
        bs = BS or DECODE_BS[rank % len(DECODE_BS)]
    x, wts, ids = T._make_inputs(
        bs, net["model_dim"], net["experts"], net["topk"], rank, 123, device
    )
    # VARY=N: cycle through N independently routed input sets so the ACTIVE
    # expert set changes from call to call, as it does in real serving (every
    # layer and every step routes differently). All repros so far reused one
    # fixed (x, wts, ids), which keeps the set of (expert, destination) pairs
    # that take part in the handshake identical on every call.
    sets = [(x, wts, ids)]
    for k in range(1, VARY):
        sets.append(
            T._make_inputs(
                bs, net["model_dim"], net["experts"], net["topk"], rank, 1000 + k, device
            )
        )

    # CFGCYCLE="2,3,4,5,8": change config_tokens from call to call, as ATOM now
    # does once config selection follows dp_metadata.max_tokens_across_dp (that
    # value moves every step). entry_count is indexed by grid_epoch_slot, which
    # is keyed on grid_mult ALONE, while generation = ticket64 // launch_grid_x
    # divides by a geometry that also depends on num_dispatch_cu and grid_x. Two
    # configs sharing a grid_mult but differing elsewhere therefore share one
    # ticket counter and divide it differently, so generation comes out wrong
    # and the strict-equality epoch gate can never be satisfied.
    cfg_cycle = [int(v) for v in os.environ.get("CFGCYCLE", "").split(",") if v]
    kw = {"config_tokens": CFG} if CFG else {}
    if rank == 0:
        print(
            f"MIN_START bs={'asym' if not BS else BS} cfg={CFG or 'natural'} "
            f"depth={DEPTH} total={TOTAL}",
            flush=True,
        )

    done = 0
    while done < TOTAL:
        for i in range(DEPTH):
            x_i, w_i, id_i = sets[(done + i) % len(sets)]
            call_kw = (
                {"config_tokens": cfg_cycle[(done + i) % len(cfg_cycle)]}
                if cfg_cycle
                else kw
            )
            moe.forward(x_i, w_i, id_i, **call_kw)
        torch.cuda.synchronize()
        done += DEPTH
        if rank == 0:
            print(f"MIN {done}/{TOTAL}", flush=True)

    if rank == 0:
        print("MIN_ALL_DONE", flush=True)
    T._cleanup()


if __name__ == "__main__":
    main()
