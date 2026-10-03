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

    # The regression: under expandable segments this call used to die either
    # exporting the signal tensor (hipIpcGetMemHandle, "invalid argument") or
    # raising "Uncached IPCBuffer has no backing tensor" on the input pool.
    init_dist_env(tp_size, rankID, local_rank=rankID)

    ca_comm = get_tp_group().device_communicator.ca_comm
    pool_mode = "none"
    if ca_comm is not None:
        buf = ca_comm._pool["input"]
        pool_mode = "raw_cached" if buf._raw_cached else "torch"
        # data_ptr is the pool's contract in every mode; the raw modes have no
        # backing tensor at all.
        assert buf.data_ptr != 0

    x = torch.full(shape, float(rankID + 1), dtype=torch.bfloat16, device=device)
    out = tensor_model_parallel_all_reduce(x).cpu()

    destroy_dist_env()
    return pool_mode, out


def test_init_dist_env(tp_size, shape, run_mode):
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
    modes = {mode for mode, _ in rets}
    for mode, out in rets:
        checkAllclose(
            ref,
            out,
            msg=f"init_dist_env allreduce: {tp_size=} mode={run_mode} pool={mode}",
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
    return {"pool_modes": sorted(modes)}


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="config input of test")
    parser.add_argument("-t", "--tp_size", type=int, default=2)
    args = parser.parse_args()

    for run_mode in ("default", "expandable", "raw_override"):
        ret = test_init_dist_env(args.tp_size, (128, 8192), run_mode)
        print(f"mode={run_mode}: {ret}")
