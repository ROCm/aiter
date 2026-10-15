# SPDX-License-Identifier: MIT
"""End-to-end four-GPU validation of the experimental M3 MoE region."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist

from minimax_m3_gluon.host import (
    allocate_peer,
    allocate_workspace,
    expert_weights,
    launch_region,
)


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    try:
        # The quantized bytes are deliberately simple; this validates tensor
        # ownership and collectives without needing model checkpoint loading.
        w13 = torch.full((129, 1536, 3072), 0x11, device="cuda", dtype=torch.uint8)
        w2 = torch.full((129, 6144, 384), 0x11, device="cuda", dtype=torch.uint8)
        s13 = torch.full((129, 1536, 192), 127, device="cuda", dtype=torch.uint8)
        s2 = torch.full((129, 6144, 24), 127, device="cuda", dtype=torch.uint8)
        weights = expert_weights(w13, s13, w2, s2)
        for batch in (1, 8, 16):
            torch.manual_seed(301 + rank)
            attn = torch.randn((batch, 6144), device="cuda", dtype=torch.bfloat16)
            residual = torch.randn_like(attn)
            gain = torch.zeros(6144, device="cuda", dtype=torch.bfloat16)
            gate = torch.randn((128, 6144), device="cuda", dtype=torch.bfloat16) * 0.01
            bias = torch.randn(128, device="cuda", dtype=torch.float32) * 0.05
            state = allocate_workspace(batch, "cuda")
            with allocate_peer(batch, rank, dist.group.WORLD) as peer:
                dist.barrier()
                out, midway = launch_region(
                    attn, residual, gain, gate, bias, weights, state,
                    peer, rank, batch, shared_weight=1.0,
                )
                torch.cuda.synchronize()
                assert state["FrontErrors"].item() == 0
                assert not state["TailErrors"].any().item()
                assert torch.equal(midway, (attn.float() + residual.float()).bfloat16())
                local = state["SortOut"].cpu()
                gathered = [None] * 4
                dist.all_gather_object(gathered, local)
                expected = torch.zeros_like(local, dtype=torch.float32)
                for value in gathered:
                    expected += value.float()
                assert torch.equal(out.cpu(), expected.bfloat16()), f"TP4 output mismatch, rank={rank}"
                if rank == 0:
                    print(f"B={batch}: integrated front + AITER MoE1/MoE2 + IPC tail OK", flush=True)
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replay_out, _ = launch_region(
                        attn, residual, gain, gate, bias, weights, state,
                        peer, rank, batch, shared_weight=1.0,
                    )
                dist.barrier()
                for _ in range(3):
                    graph.replay()
                    torch.cuda.synchronize()
                    assert not state["TailErrors"].any().item()
                    gathered = [None] * 4
                    dist.all_gather_object(gathered, state["SortOut"].cpu())
                    expected = torch.zeros_like(gathered[0], dtype=torch.float32)
                    for value in gathered:
                        expected += value.float()
                    assert torch.equal(replay_out.cpu(), expected.bfloat16())
                    dist.barrier()
                if rank == 0:
                    print(f"B={batch}: TP4 graph replay x3 OK", flush=True)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                dist.barrier()
                start.record()
                for _ in range(64):
                    graph.replay()
                end.record()
                end.synchronize()
                elapsed_us = start.elapsed_time(end) * 1000 / 64
                latencies = [None] * 4
                dist.all_gather_object(latencies, elapsed_us)
                if rank == 0:
                    print(
                        f"B={batch}: full-region graph GPU time "
                        f"max-rank={max(latencies):.2f}us/replay",
                        flush=True,
                    )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
