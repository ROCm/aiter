# SPDX-License-Identifier: MIT
"""Task 2 validation: push-group dispatch lands tokens grouped per local expert.

Run (single rank is enough; world_size=1 => every expert is local, dest_pe=0):

    FLYDSL_GPU_ARCH=$(python -c \
      "from aiter import get_gfx;print(get_gfx())") \
    torchrun --standalone --nproc_per_node=1 \
      /app/aiter/op_tests/flydsl_tests/test_push_group_dispatch.py

push-group is the explicit ``push_group=True`` config switch (no env var).

Checks (byte-exact, no tolerance):
  * pg_running[e] == number of (token,slot) routed to local expert e
  * every grouped row e*CAP .. e*CAP+count maps (via tis) to a source token that
    was routed to e, and disp_out[row] equals that source token's embedding bits
  * the multiset of source tokens in expert e's group == routed set for e
"""
import os

import torch
import torch.distributed as dist

from mori.cco import Communicator
from mori.tensor_utils import from_gpu_ptr
from aiter.ops.flydsl.dispatch_combine_v2 import (
    EpDispatchCombineConfig,
    EpDispatchCombineOp,
)


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    torch.manual_seed(1234 + rank)

    hidden = 512
    n_tok = 8
    num_experts_per_rank = 4
    topk = 2
    max_tok = n_tok

    uid = Communicator.get_unique_id() if rank == 0 else None
    objs = [uid]
    dist.broadcast_object_list(objs, src=0)
    uid = objs[0]
    comm = Communicator.init(world, rank, uid)

    cfg = EpDispatchCombineConfig(
        rank=rank,
        world_size=world,
        hidden_dim=hidden,
        max_num_inp_token_per_rank=max_tok,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=topk,
        data_type=torch.bfloat16,
        combine_mode="gather",
        push_group=True,
    )
    assert cfg.push_group, "push_group switch not set on config"
    op = EpDispatchCombineOp(cfg, comm)
    comm.barrier()

    num_experts = num_experts_per_rank * world
    x = torch.randn(n_tok, hidden, dtype=torch.bfloat16, device=dev)
    # distinct experts per token
    ids = torch.empty(n_tok, topk, dtype=torch.int32, device=dev)
    for t in range(n_tok):
        perm = torch.randperm(num_experts, device=dev)[:topk]
        ids[t] = perm.to(torch.int32)
    wts = torch.rand(n_tok, topk, dtype=torch.float32, device=dev)

    op.dispatch(x, wts, None, ids, return_routing=True)
    comm.barrier()
    torch.cuda.synchronize()

    cap = cfg.effective_cap_per_expert
    rrows = op._pg_recv_rows
    disp_out = from_gpu_ptr(
        op.arena.local_ptr("disp_out"), (rrows, hidden), torch.bfloat16
    )
    tis = from_gpu_ptr(
        op.arena.local_ptr("recv_to_src_token"), (rrows,), torch.int32
    )
    pg_running = from_gpu_ptr(
        op.arena.local_ptr("pg_running"), (num_experts_per_rank,), torch.int32
    ).cpu()

    # reference: this rank (world=1) owns experts [0, num_experts_per_rank)
    ids_cpu = ids.cpu()
    x_bits = x.view(torch.int16).cpu()
    ok = True
    for e in range(num_experts_per_rank):
        routed = [
            t
            for t in range(n_tok)
            for k in range(topk)
            if int(ids_cpu[t, k]) == e
        ]
        cnt = int(pg_running[e])
        if cnt != len(routed):
            print(f"[FAIL] expert {e}: pg_running={cnt} expected={len(routed)}")
            ok = False
            continue
        got_src = []
        for r in range(cnt):
            row = e * cap + r
            src = int(tis[row].item())  # rank*max_tok + src_tok; rank=0 -> src_tok
            src_tok = src % max_tok
            got_src.append(src_tok)
            if not torch.equal(
                disp_out[row].view(torch.int16).cpu(), x_bits[src_tok]
            ):
                print(f"[FAIL] expert {e} row {row}: embedding mismatch src={src_tok}")
                ok = False
        if sorted(got_src) != sorted(routed):
            print(
                f"[FAIL] expert {e}: grouped src {sorted(got_src)} != routed {sorted(routed)}"
            )
            ok = False

    print("[PASS] push-group dispatch grouped landing" if ok else "[FAIL] see above")
    dist.destroy_process_group()
    assert ok


if __name__ == "__main__":
    main()
