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
state, agrees on the result across ranks, and refuses a captured input that
cannot be exported.

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

# scenario -> (expandable_segments at __init__, which ranks flip them for
# capture, expected registered path inside capture()).
#   keep:    left on, the copy-in path is kept.
#   toggle:  the host turns them off for capture, the registered path is used.
#   mixed:   only rank 0 turns them off; ranks must agree, so all use copy-in.
#   reverse: off at __init__, on at capture; the registered path is refused.
#   stale:   turned off for capture, but the input was allocated before, so it
#            is VMM-backed; the captured collective must raise, not abort, and
#            leave the process usable.
SCENARIOS = {
    "keep": (True, "none", False),
    "toggle": (True, "all", True),
    "mixed": (True, "rank0", False),
    "reverse": (False, "all", False),
    "stale": (True, "all", True),
}


def _set_expandable_segments(enabled: bool) -> None:
    """What a host (e.g. vLLM) does around capture. Keep the environment in
    step with the live setting: _expandable_segments_enabled() falls back to
    parsing it when the allocator snapshot cannot be read."""
    value = "True" if enabled else "False"
    torch._C._accelerator_setAllocatorSettings(f"expandable_segments:{value}")
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = f"expandable_segments:{value}"


def _worker(tp_size, rankID, scenario, shape):
    at_init, flip_ranks, _ = SCENARIOS[scenario]
    # Programmatic, not just the environment: importing aiter has already
    # initialized CUDA in this process, so the allocator config is parsed and
    # a late PYTORCH_HIP_ALLOC_CONF write would be ignored.
    _set_expandable_segments(at_init)

    from aiter.dist.communication_op import tensor_model_parallel_all_reduce
    from aiter.dist.device_communicators.custom_all_reduce import (
        _expandable_segments_enabled,
    )
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
    latched = ca_comm.enable_register_for_capturing
    requested = getattr(ca_comm, "_register_for_capturing_requested", None)
    # The allocator snapshot reports whether expandable_segments took effect.
    at_init_seen = _expandable_segments_enabled()

    flip = flip_ranks == "all" or (flip_ranks == "rank0" and rankID == 0)
    fill = float(rankID + 1)
    x = None
    if scenario == "stale":
        x = torch.full(shape, fill, dtype=torch.bfloat16, device=device)

    graph = torch.cuda.CUDAGraph()
    registered_during_capture = None
    error = None
    out_t = None
    if flip:
        _set_expandable_segments(not at_init)
    at_capture_seen = _expandable_segments_enabled()
    try:
        # capture() must wrap the graph, as graph_capture() does: its exit
        # flushes the graph buffers, which cannot run on a capturing stream.
        # Allocate inside the toggled window, like a graph-pool activation.
        with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
            registered_during_capture = ca_comm.enable_register_for_capturing
            if x is None:
                x = torch.full(shape, fill, dtype=torch.bfloat16, device=device)
            out_t = tensor_model_parallel_all_reduce(x)
    except RuntimeError as e:
        if scenario != "stale":
            raise
        error = str(e)
    finally:
        if flip:
            _set_expandable_segments(at_init)

    if error is None:
        graph.replay()
    else:
        # A refused capture must leave the process usable: no sticky HIP error
        # for the next launch to report, and a working communicator.
        out_t = tensor_model_parallel_all_reduce(
            torch.full(shape, fill, dtype=torch.bfloat16, device=device)
        )
    torch.cuda.synchronize()
    out = out_t.cpu()

    restored = ca_comm.enable_register_for_capturing
    destroy_dist_env()
    return {
        "latched": latched,
        "requested": requested,
        "seen": (at_init_seen, at_capture_seen),
        "flip": flip,
        "during_capture": registered_during_capture,
        "restored": restored,
        "disabled": disabled,
        "error": error,
        "out": out,
    }


def test_capture_registration(tp_size, shape, scenario, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    pool = Pool(processes=tp_size)
    rets = [
        pool.apply_async(_worker, args=(tp_size, i, scenario, shape))
        for i in range(tp_size)
    ]
    pool.close()
    # A rank that dies leaves its result pending forever; fail, don't hang.
    rets = [r.get(timeout=600) for r in rets]
    pool.join()

    at_init, _, expected = SCENARIOS[scenario]
    tag = f"{tp_size=} {scenario=}"
    disabled = all(r["disabled"] for r in rets)
    # No veto to exercise: gfx1250 and the VMM transport never register
    # captured inputs, and a platform may ignore expandable_segments. Every
    # rank must also see exactly the allocator state the scenario set up.
    exercised = not disabled and all(
        r["requested"] and r["seen"] == (at_init, at_init != r["flip"]) for r in rets
    )
    raises = scenario == "stale" and exercised

    # sum over ranks of full(rank+1) = n(n+1)/2; for "stale", from the eager
    # all-reduce run after the refused capture.
    ref = torch.full(shape, float(tp_size * (tp_size + 1) // 2), dtype=torch.bfloat16)
    for i, r in enumerate(rets):
        if raises:
            assert r["error"] and "cannot be IPC-exported" in r["error"], (
                f"{tag} rank {i}: a VMM-backed input on the registered path "
                f"must raise a clear error, got {r['error']!r}"
            )
        else:
            assert r["error"] is None, f"{tag} rank {i}: {r['error']}"
        checkAllclose(ref, r["out"], msg=f"allreduce: {tag} rank={i}")

    if disabled:
        return {"disabled": True}
    if not exercised:
        logger.warning(
            f"{tag}: expandable_segments veto not exercised on this platform "
            f"(requested={[r['requested'] for r in rets]}, "
            f"seen={[r['seen'] for r in rets]}); checked the result only."
        )
        return {"veto_exercised": False}

    for i, r in enumerate(rets):
        assert r["latched"] is (not at_init), (
            f"{tag} rank {i}: __init__ must veto the registered path exactly "
            f"when expandable_segments are on; saw {r['latched']}"
        )
        # "mixed" also checks agreement: rank 0 alone could have registered.
        assert r["during_capture"] is expected, (
            f"{tag} rank {i}: expected enable_register_for_capturing="
            f"{expected} inside capture(), saw {r['during_capture']}"
        )
        assert r["restored"] is r["latched"], (
            f"{tag} rank {i}: the capture-time decision must not leak out of "
            "capture()"
        )
    return {"during_capture": expected, "raised": raises}


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="config input of test")
    parser.add_argument(
        "-t",
        "--tp_size",
        type=int,
        nargs="*",
        default=None,
        help="TP sizes (default: 2 3 4 8, capped at the visible GPU count; "
        "3 exercises a disabled communicator)",
    )
    parser.add_argument(
        "-s",
        "--scenario",
        choices=list(SCENARIOS),
        nargs="*",
        default=list(SCENARIOS),
    )
    args = parser.parse_args()

    n_gpu = torch.cuda.device_count()
    tp_sizes = args.tp_size or [tp for tp in (2, 3, 4, 8) if tp <= n_gpu]
    port = 49375
    for tp_size in tp_sizes:
        for scenario in args.scenario:
            # A fresh port per run: the previous store may still hold its own.
            ret = test_capture_registration(tp_size, (128, 8192), scenario, port)
            port += 1
            print(f"tp_size={tp_size} scenario={scenario}: {ret}")
