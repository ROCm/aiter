# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The expandable_segments veto must be re-taken when capture() is entered.

Under PYTORCH_HIP_ALLOC_CONF=expandable_segments:True the caching allocator
hands out VMM-backed pointers, which hipIpcGetMemHandle cannot export (#4174).
CustomAllreduce therefore refuses the registered capture path and stages every
input through a device-to-device copy.

That decision used to be latched once in __init__. A host that disables
expandable segments only for the duration of CUDA graph capture -- so that the
graph pool's activations *are* exportable -- got no benefit from it, because
the decision had already been made. capture() now re-reads the live allocator
state, so the registered path is picked up.

Two axes that no existing test covers together: test_init_dist_env.py sets
expandable segments but captures no graph, and test_custom_allreduce.py
captures a graph but never touches the allocator config.
"""

import argparse
import logging
import os
from multiprocessing import Pool, freeze_support, set_start_method

import torch

from aiter.test_common import checkAllclose

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)


def _set_expandable_segments(enabled: bool) -> None:
    """What a host (e.g. vLLM) does around capture. Keep the environment in
    step with the live setting: _expandable_segments_enabled() falls back to
    parsing it when the allocator snapshot cannot be read, and a rank that
    read a stale value would pick a different path from its peers."""
    value = "True" if enabled else "False"
    torch._C._accelerator_setAllocatorSettings(f"expandable_segments:{value}")
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = f"expandable_segments:{value}"


def _worker(tp_size, rankID, toggle_for_capture, shape):
    # Programmatic, not just the environment: importing aiter has already
    # initialized CUDA in this process, so the allocator config is parsed and
    # a late PYTORCH_HIP_ALLOC_CONF write would be ignored.
    _set_expandable_segments(True)

    from aiter.dist.communication_op import tensor_model_parallel_all_reduce
    from aiter.dist.parallel_state import get_tp_group, graph_capture
    from aiter.ops.communication import destroy_dist_env, init_dist_env

    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    init_dist_env(tp_size, rankID, local_rank=rankID)

    ca_comm = get_tp_group().device_communicator.ca_comm
    assert ca_comm is not None, "custom all-reduce not available"
    # A disabled communicator (e.g. an unsupported world size) is still
    # entered by graph_capture(); capture must keep working there.
    disabled = ca_comm.disabled
    # The __init__ veto must still fire: expandable segments are on right now.
    latched = ca_comm.enable_register_for_capturing
    requested = getattr(ca_comm, "_register_for_capturing_requested", None)

    graph = torch.cuda.CUDAGraph()
    registered_during_capture = None
    if toggle_for_capture:
        _set_expandable_segments(False)
    try:
        # capture() must wrap the graph, as graph_capture() does: its exit
        # flushes the graph buffers, which cannot run on a capturing stream.
        # Allocate inside the toggled window, like a graph-pool activation.
        with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
            registered_during_capture = ca_comm.enable_register_for_capturing
            x = torch.full(
                shape, float(rankID + 1), dtype=torch.bfloat16, device=device
            )
            out_t = tensor_model_parallel_all_reduce(x)
    finally:
        if toggle_for_capture:
            _set_expandable_segments(True)

    graph.replay()
    torch.cuda.synchronize()
    out = out_t.cpu()

    restored = ca_comm.enable_register_for_capturing
    destroy_dist_env()
    return {
        "latched": latched,
        "requested": requested,
        "during_capture": registered_during_capture,
        "restored": restored,
        "disabled": disabled,
        "out": out,
    }


def test_capture_registration(tp_size, shape, toggle_for_capture):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49375"
    pool = Pool(processes=tp_size)
    rets = [
        pool.apply_async(_worker, args=(tp_size, i, toggle_for_capture, shape))
        for i in range(tp_size)
    ]
    pool.close()
    # A rank that dies leaves its result pending forever; fail, don't hang.
    rets = [r.get(timeout=600) for r in rets]
    pool.join()

    # sum over ranks of full(rank+1) = n(n+1)/2
    ref = torch.full(shape, float(tp_size * (tp_size + 1) // 2), dtype=torch.bfloat16)
    for i, r in enumerate(rets):
        checkAllclose(
            ref,
            r["out"],
            msg=f"captured allreduce: {tp_size=} toggle={toggle_for_capture} rank={i}",
        )

    if all(r["disabled"] for r in rets):
        return {"disabled": True}
    # No veto to re-take: gfx1250 and the VMM transport never veto, and a
    # platform that ignores expandable_segments never sets it.
    if not all(r["requested"] for r in rets) or any(r["latched"] for r in rets):
        logger.warning(
            "expandable_segments veto not exercised on this platform "
            f"(requested={[r['requested'] for r in rets]}, "
            f"latched={[r['latched'] for r in rets]}); checked the result only."
        )
        return {"veto_exercised": False}

    for i, r in enumerate(rets):
        assert r["latched"] is False, (
            "expandable_segments is on at __init__, so the registered capture "
            f"path must be vetoed there; rank {i} saw {r['latched']}"
        )
        assert r["during_capture"] is toggle_for_capture, (
            f"rank {i}: expected enable_register_for_capturing="
            f"{toggle_for_capture} inside capture(), saw {r['during_capture']}"
        )
        assert (
            r["restored"] is False
        ), f"rank {i}: the capture-time decision must not leak out of capture()"

    # Every rank must agree: the registered/unregistered choice changes the
    # kernel algorithm and the graph-buffer count, so a split is a collective
    # mismatch, not a local inefficiency.
    decisions = {r["during_capture"] for r in rets}
    assert len(decisions) == 1, f"ranks disagreed on the capture path: {rets}"
    return {"during_capture": decisions.pop()}


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="config input of test")
    parser.add_argument("-t", "--tp_size", type=int, default=2)
    args = parser.parse_args()

    # toggle=False: expandable segments stay on, the copy-in path is kept.
    # toggle=True:  the host turns them off for capture, the registered
    #               (copy-free) path is picked up.
    # -t 3 exercises a disabled communicator (unsupported world size).
    for toggle in (False, True):
        ret = test_capture_registration(args.tp_size, (128, 8192), toggle)
        print(f"toggle_for_capture={toggle}: {ret}")
