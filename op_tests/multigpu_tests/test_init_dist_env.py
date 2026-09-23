# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""init_dist_env must come up whatever allocation mode backs the IPC input pool.

Regression test for #4921. Under PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
init_dist_env used to fail twice over:

  * it registered a torch.zeros "signal" tensor, whose VMM-backed pointer
    hipIpcGetMemHandle rejects (custom_all_reduce.cu:417), and
  * it read `_pool["input"].tensor`, which raises by design for the raw_cached
    (plain-hipMalloc) input pool that expandable segments selects.

So every raw_cached configuration died at distributed init. Both allocator
modes are exercised here and share one allreduce-correctness check.

The "dirty" mode covers a second, separate bug in the same area: _broadcast_ipc
used to pin the IPC offset to 0, which is only correct when the pool pointer is
an allocation base. With a fresh caching allocator it always is, so a first
engine is fine; once the allocator holds a cached large segment (a co-resident
second engine, or simply an earlier alloc/free) the pool lands as a sub-block
and peers addressed the segment base instead -- wrong allreduce results, no
error. Dirtying the allocator before init_dist_env reproduces that shape.

The existing test_custom_allreduce.py does not cover this: it performs its own
init and never runs init_dist_env.
"""

import argparse
import logging
import os
from multiprocessing import Pool, freeze_support, set_start_method

import torch

from aiter.test_common import checkAllclose

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)

GiB = 1024 * 1024 * 1024


def _worker(tp_size, rankID, mode, shape):
    # Must precede CUDA init in this process: the allocator reads the setting
    # once, when the context comes up.
    if mode == "expandable":
        os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
    elif mode == "raw_override":
        os.environ["AITER_CUSTOM_AR_RAW_INPUT_POOL"] = "1"

    from aiter.dist.communication_op import tensor_model_parallel_all_reduce
    from aiter.dist.parallel_state import get_tp_group
    from aiter.ops.communication import destroy_dist_env, init_dist_env

    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)

    keep = None
    if mode == "dirty":
        # Make the IPC pool that init_dist_env allocates next come back as a
        # sub-block of an existing allocation rather than as its own segment:
        # hold a few GiB of "weights", free a large activation segment, and
        # keep one live block at its head. This is the shape a loaded,
        # co-resident engine produces incidentally.
        #
        # Scale matters. A 512 MiB segment is not enough -- the allocations
        # init_dist_env makes on the way to the pool consume the remainder and
        # the pool still gets a fresh segment (measured: offset 0). Multi-GiB
        # reproduces reliably.
        keep = [torch.empty(GiB, dtype=torch.uint8, device=device) for _ in range(6)]
        act = torch.empty(4 * GiB, dtype=torch.uint8, device=device)
        del act
        keep.append(torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device))

    # The regression: under expandable segments this call used to die either
    # exporting the signal tensor (hipIpcGetMemHandle, "invalid argument") or
    # raising "Uncached IPCBuffer has no backing tensor" on the input pool.
    init_dist_env(tp_size, rankID, local_rank=rankID)

    ca_comm = get_tp_group().device_communicator.ca_comm
    pool_mode = "none"
    ipc_offset = 0
    if ca_comm is not None:
        buf = ca_comm._pool["input"]
        pool_mode = "raw_cached" if buf._raw_cached else "torch"
        # data_ptr is the pool's contract in every mode; the raw modes have no
        # backing tensor at all.
        assert buf.data_ptr != 0
        # Report the pool's offset within its allocation so a run that did not
        # actually land on a sub-block is distinguishable from one that did and
        # handled it. The allreduce check below is what proves correctness.
        from aiter.dist.device_communicators.custom_all_reduce import _ipc_base_ptr

        ipc_offset = buf.data_ptr - _ipc_base_ptr(buf.data_ptr)

    x = torch.full(shape, float(rankID + 1), dtype=torch.bfloat16, device=device)
    out = tensor_model_parallel_all_reduce(x).cpu()

    destroy_dist_env()
    del keep
    return pool_mode, ipc_offset, out


def test_init_dist_env(tp_size, shape, run_mode):
    if run_mode == "dirty":
        free = min(
            torch.cuda.mem_get_info(torch.device(f"cuda:{i}"))[0]
            for i in range(tp_size)
        )
        if free < 16 * GiB:
            logger.warning(
                "skipping dirty mode: needs ~16 GiB free per GPU, have %.1f GiB",
                free / GiB,
            )
            return {"pool_modes": [], "ipc_offsets": [], "skipped": True}
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49374"
    pool = Pool(processes=tp_size)
    rets = [
        pool.apply_async(_worker, args=(tp_size, i, run_mode, shape))
        for i in range(tp_size)
    ]
    pool.close()
    pool.join()
    rets = [r.get() for r in rets]

    # sum over ranks of full(rank+1) = n(n+1)/2
    ref = torch.full(shape, float(tp_size * (tp_size + 1) // 2), dtype=torch.bfloat16)
    modes = {mode for mode, _, _ in rets}
    offsets = [off for _, off, _ in rets]
    for mode, off, out in rets:
        checkAllclose(
            ref,
            out,
            msg=(
                f"init_dist_env allreduce: {tp_size=} mode={run_mode} "
                f"pool={mode} ipc_offset={off}"
            ),
        )
    if run_mode == "raw_override":
        assert modes == {
            "raw_cached"
        }, f"AITER_CUSTOM_AR_RAW_INPUT_POOL did not select the raw pool: {modes}"
    if run_mode == "expandable" and modes == {"torch"}:
        # The allocator snapshot is authoritative; a platform that does not
        # honor expandable segments falls back to the torch pool, and this run
        # then did not exercise the raw_cached path.
        logger.warning(
            "expandable_segments requested but the input pool is not raw_cached; "
            "raw path NOT exercised on this platform"
        )
    if run_mode == "dirty" and modes == {"torch"} and not any(offsets):
        # Nothing was exercised: this allocator handed out a fresh segment
        # anyway, so the sub-block path never came up.
        logger.warning(
            "dirty mode did not produce a sub-block pool (all offsets 0); "
            "the IPC offset path was NOT exercised on this platform"
        )
    return {"pool_modes": sorted(modes), "ipc_offsets": offsets}


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="config input of test")
    parser.add_argument("-t", "--tp_size", type=int, default=2)
    args = parser.parse_args()

    for run_mode in ("default", "dirty", "expandable", "raw_override"):
        ret = test_init_dist_env(args.tp_size, (128, 8192), run_mode)
        print(f"mode={run_mode}: {ret}")
