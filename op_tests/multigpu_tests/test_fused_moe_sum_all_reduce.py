# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness test for the fused expert-sum + all-reduce op.

Reference : aiter.moe_sum(stack) -> tensor_model_parallel_all_reduce(...)
Candidate : tensor_model_parallel_fused_moe_sum_all_reduce(stack)

The candidate folds the topk expert weighted-sum into the all-reduce input
stage; it must match the reference sum-then-reduce path within fp tolerance.

Run manually (2 GPUs):
    HIP_VISIBLE_DEVICES="0,7" python op_tests/multigpu_tests/test_fused_moe_sum_all_reduce.py -t 2
Run manually (4 GPUs):
    HIP_VISIBLE_DEVICES="0,1,2,7" python op_tests/multigpu_tests/test_fused_moe_sum_all_reduce.py -t 4
"""

import argparse
import itertools
import logging
import os
from multiprocessing import Pool, freeze_support, set_start_method
from typing import Optional

import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.dist.communication_op import (
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_fused_moe_sum_all_reduce,
)
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.test_common import checkAllclose

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)


def _worker(
    tp_size,
    pp_size,
    rank,
    x,
    distributed_init_method: Optional[str] = None,
):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rank,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)
    x = x.to(device)

    if tp_size > 1:
        group = get_tp_group().device_group
        dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    try:
        # Candidate: fused expert-sum + all-reduce.
        fused = tensor_model_parallel_fused_moe_sum_all_reduce(x)

        # Reference: standalone moe_sum then a separate all-reduce.
        out_shape = x.shape[:-2] + x.shape[-1:]
        summed = torch.empty(out_shape, dtype=x.dtype, device=device)
        aiter.moe_sum(x.contiguous(), summed)
        ref = tensor_model_parallel_all_reduce(summed)

        return fused.cpu(), ref.cpu()
    finally:
        if dist.is_initialized():
            destroy_model_parallel()
            destroy_distributed_environment()
            torch.cuda.empty_cache()


def run_case(tp_size, pp_size, m, topk, n, dtype, distributed_init_method=None):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49375"
    torch.manual_seed(0)
    # Same input on every rank; the all-reduce sums identical contributions.
    x = torch.randn((m, topk, n), dtype=dtype) * 0.1

    with Pool(processes=tp_size) as pool:
        rets = [
            pool.apply_async(
                _worker,
                args=(tp_size, pp_size, rank, x, distributed_init_method),
            )
            for rank in range(tp_size)
        ]
        rets = [r.get() for r in rets]

    # Host reference: sum over topk then across ranks (identical x per rank).
    host_ref = (x.float().sum(dim=-2) * tp_size).to(dtype)

    atol = 1e-2 if dtype != torch.bfloat16 else 5e-2
    rtol = atol
    max_err = 0.0
    max_ref_err = 0.0
    for rank, (fused, ref) in enumerate(rets):
        assert fused.shape == (m, n), f"bad shape {fused.shape}"
        max_err = max(
            max_err,
            checkAllclose(
                ref.to(fused),
                fused,
                msg=f"fused vs sum-then-AR: {tp_size=} {m=} {topk=} {n=} {dtype=} rank={rank}",
                atol=atol,
                rtol=rtol,
            ),
        )
        max_ref_err = max(
            max_ref_err,
            checkAllclose(
                host_ref,
                fused,
                msg=f"fused vs host ref: {tp_size=} {m=} {topk=} {n=} {dtype=} rank={rank}",
                atol=atol,
                rtol=rtol,
            ),
        )
    return {
        "tp_size": tp_size,
        "m": m,
        "topk": topk,
        "n": n,
        "dtype": str(dtype),
        "err_vs_sum_then_ar": max_err,
        "err_vs_host_ref": max_ref_err,
    }


try:
    import pytest

    @pytest.mark.parametrize("topk", [2, 8])
    @pytest.mark.parametrize("dtype_key", ["bf16", "fp16"])
    def test_fused_moe_sum_all_reduce_tp2(topk, dtype_key):
        if torch.cuda.device_count() < 2:
            pytest.skip(f"requires >= 2 GPUs (have {torch.cuda.device_count()})")
        ret = run_case(
            tp_size=2,
            pp_size=1,
            m=17,
            topk=topk,
            n=6144,
            dtype=dtypes.d_dtypes[dtype_key],
            distributed_init_method=get_distributed_init_method(
                get_ip(), get_open_port()
            ),
        )
        assert ret["err_vs_sum_then_ar"] < 5e-2, ret
        assert ret["err_vs_host_ref"] < 5e-2, ret

except ImportError:
    pass


l_dtype = ["bf16", "fp16"]
l_shape = [(17, 6144), (128, 6144), (1, 6144)]
l_topk = [2, 8]

parser = argparse.ArgumentParser(description="fused moe_sum + all_reduce test")
parser.add_argument("-t", "--tp", type=int, default=2, help="tp size, e.g. -t 2")
parser.add_argument("-p", "--pp", type=int, default=1, help="pp size")
parser.add_argument("-d", "--dtype", type=str, choices=l_dtype, default=None)


if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    dtypes_to_run = [args.dtype] if args.dtype else l_dtype
    rows = []
    for dkey, (m, n), topk in itertools.product(dtypes_to_run, l_shape, l_topk):
        ret = run_case(
            tp_size=args.tp,
            pp_size=args.pp,
            m=m,
            topk=topk,
            n=n,
            dtype=dtypes.d_dtypes[dkey],
            distributed_init_method=get_distributed_init_method(
                get_ip(), get_open_port()
            ),
        )
        rows.append(ret)
        logger.info("%s", ret)
    try:
        import pandas as pd

        print(pd.DataFrame(rows).to_markdown(index=False))
    except ImportError:
        for r in rows:
            print(r)
