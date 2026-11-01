# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Graph-buffer registration must stay in step across ranks.

On the registered capture path an fp32/fp16 all-reduce records both its input
and its output (bf16 records the input only). After capture,
flush_graph_buffers exchanges each rank's list, and register_graph_buffers
pairs buffer i with every peer's buffer i.

register_graph_buffers used to also add each captured output to the map of
eagerly registered outputs. A later capture consults that map before recording,
so a rank whose allocator handed out the same address again skipped the buffer
while the other ranks recorded theirs. The lists then had different lengths,
and peers opened IPC handles that belonged to other buffers. vLLM hit this with
DSpark on Muse Spark, whose MoE reduces in fp32: the draft graphs, captured
after the target graphs, came up 4 buffers short on one of 8 ranks, and
hipIpcOpenMemHandle failed with "invalid argument". Had the counts happened to
match, the stale entry would have pointed the rank at its peers' old buffers.

Scenarios (all on the second of two captures):
  reuse: rank 0 writes into the first capture's output, the other ranks into a
         fresh tensor. Every rank must record 2 buffers, register, and replay
         both graphs to the exact sum.
  extra: rank 0 captures one more all-reduce than the others.
  none:  rank 0 captures no all-reduce.
The last two must raise the same error on every rank at flush, without hanging
the ranks that recorded more buffers.
"""

import argparse
import logging
import os
from multiprocessing import Pool, freeze_support, set_start_method

import torch

from aiter.dist.utils import get_open_port
from aiter.test_common import assertAllclose

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)

SCENARIOS = ("reuse", "extra", "none")


def _worker(tp_size, rankID, scenario, shape):
    torch._C._accelerator_setAllocatorSettings("expandable_segments:False")
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:False"

    from aiter.dist.parallel_state import get_tp_group, graph_capture
    from aiter.ops.communication import destroy_dist_env, init_dist_env

    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    init_dist_env(tp_size, rankID, local_rank=rankID)
    try:
        ca_comm = get_tp_group().device_communicator.ca_comm
        if ca_comm is None or ca_comm.disabled:
            return {"skipped": "custom all-reduce is not available"}
        if not ca_comm.enable_register_for_capturing:
            return {"skipped": "the registered capture path is off"}

        counts = []
        count_fn = ca_comm._pool._graph_count_fn

        def recording_count_fn(ptr):
            counts.append(count_fn(ptr))
            return counts[-1]

        ca_comm._pool._graph_count_fn = recording_count_fn

        def capture(calls):
            graph = torch.cuda.CUDAGraph()
            with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
                for x, out in calls:
                    ca_comm.all_reduce(x, out=out, registered_input=True)
            return graph

        def full(scale):
            return torch.full(
                shape, float(scale * (rankID + 1)), dtype=torch.float32, device=device
            )

        x1 = full(1)
        out1 = torch.empty_like(x1)
        graph1 = capture([(x1, out1)])

        x2 = full(2)
        out2 = out1 if scenario == "reuse" and rankID == 0 else torch.empty_like(x2)
        calls = [(x2, out2)]
        if rankID == 0 and scenario == "extra":
            calls.append((full(3), torch.empty_like(x2)))
        elif rankID == 0 and scenario == "none":
            calls = []
        try:
            graph2 = capture(calls)
        except RuntimeError as e:
            return {"counts": counts, "error": str(e)}

        graph2.replay()
        torch.cuda.synchronize()
        got2 = out2.cpu()
        graph1.replay()
        torch.cuda.synchronize()
        got1 = out1.cpu()
        return {"counts": counts, "out1": got1, "out2": got2}
    finally:
        destroy_dist_env()


def test_graph_buffer_reuse(tp_size, shape, scenario):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(get_open_port())
    pool = Pool(processes=tp_size)
    rets = [
        pool.apply_async(_worker, args=(tp_size, i, scenario, shape))
        for i in range(tp_size)
    ]
    pool.close()
    rets = [r.get(timeout=600) for r in rets]
    pool.join()

    tag = f"{tp_size=} {scenario=}"
    if any("skipped" in r for r in rets):
        logger.warning(f"{tag}: skipped ({rets[0].get('skipped')})")
        return {"skipped": True}

    if scenario != "reuse":
        for i, r in enumerate(rets):
            assert r.get("error") and "different numbers" in r["error"], (
                f"{tag} rank {i}: a rank-divergent capture must raise on every "
                f"rank, got {r.get('error')!r}"
            )
        return {"raised": True}

    for i, r in enumerate(rets):
        assert (
            "error" not in r
        ), f"{tag} rank {i}: the second capture must register: {r['error']}"
        assert r["counts"] == [
            2,
            2,
        ], f"{tag} rank {i}: recorded {r['counts']} graph buffers, expected [2, 2]"
    total = tp_size * (tp_size + 1) // 2
    for i, r in enumerate(rets):
        for name, scale in (("out1", 1), ("out2", 2)):
            ref = torch.full(shape, float(scale * total), dtype=torch.float32)
            assertAllclose(
                ref,
                r[name],
                rtol=0,
                atol=0,
                tol_err_ratio=0,
                msg=f"allreduce {name}: {tag} rank={i}",
            )
    return {"ok": True}


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="config input of test")
    parser.add_argument(
        "-t",
        "--tp_size",
        type=int,
        nargs="*",
        default=None,
        help="TP sizes (default: 2 4 8, capped at the visible GPU count)",
    )
    parser.add_argument(
        "-s", "--scenario", choices=SCENARIOS, nargs="*", default=list(SCENARIOS)
    )
    args = parser.parse_args()

    n_gpu = torch.cuda.device_count()
    tp_sizes = args.tp_size or [tp for tp in (2, 4, 8) if tp <= n_gpu]
    for tp_size in tp_sizes:
        for scenario in args.scenario:
            ret = test_graph_buffer_reuse(tp_size, (128, 8192), scenario)
            print(f"tp_size={tp_size} scenario={scenario}: {ret}")
