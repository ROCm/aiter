# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run with torchrun --nproc-per-node=4 -m pytest on ROCm."""

import os

import pytest
import torch
import torch.distributed as dist


@pytest.fixture(scope="module")
def collective():
    if not torch.version.hip or int(os.environ.get("WORLD_SIZE", "0")) != 4:
        pytest.skip("requires four ROCm ranks launched by torchrun")
    from aiter import destroy_dist_env
    from aiter.dist.parallel_state import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(2)
    dist.init_process_group("nccl")
    init_distributed_environment(world_size=4, rank=rank, local_rank=rank)
    initialize_model_parallel(tensor_model_parallel_size=4)
    group = get_tp_group()
    ca = group.device_communicator.ca_comm
    assert ca is not None and not ca.disabled
    yield rank, ca
    torch.cuda.synchronize()
    group.barrier()
    torch.cuda.synchronize()
    destroy_dist_env()
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "rows,width,dtype,use_new",
    [
        (1, 7168, torch.bfloat16, True),
        (4, 8192, torch.bfloat16, True),
        (17, 7168, torch.bfloat16, True),
        (97, 7168, torch.bfloat16, True),
        (512, 7168, torch.bfloat16, True),
        (512, 7168, torch.bfloat16, False),
        (1025, 88, torch.bfloat16, True),
        (97, 7168, torch.float16, True),
        (512, 7168, torch.float32, True),
    ],
)
def test_rank_order_is_independent_of_row_and_receiver(
    collective, rows, width, dtype, use_new
):
    from aiter.dist.parallel_state import graph_capture

    rank, ca = collective
    # Captured BF16 partials straddle a rounding midpoint after FP32 summation.
    partials = torch.tensor(
        [
            [1.1496013030409813e-9, 5.966285243630409e-10, 1e-7],
            [-0.01531982421875, -0.00616455078125, 1.0],
            [-0.006622314453125, -0.0029449462890625, -1.0],
            [-0.006072998046875, -0.0007476806640625, -1e-6],
        ],
        dtype=dtype,
        device="cpu",
    ).repeat(1, (width + 2) // 3)[:, :width]
    x = torch.empty((rows, width), dtype=dtype, device=f"cuda:{rank}")
    x.copy_(partials[rank].expand(rows, width))
    assert ca.should_custom_ar(x)
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
        out = ca.custom_all_reduce(x, use_new=use_new)
    for factor in (1.0, -1.0, 0.5):
        changed = (partials.float() * factor).to(dtype)
        reference = changed[0].float()
        for peer in range(1, 4):
            reference = reference + changed[peer].float()
        reference = reference.to(dtype).expand(rows, width)
        x.copy_(changed[rank].expand(rows, width))
        eager = ca.custom_all_reduce(x, use_new=use_new)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(eager.cpu(), reference)
        assert torch.equal(out.cpu(), reference)
    # Retain peer IPC storage until every parameterized graph is finished.
    retained = getattr(
        test_rank_order_is_independent_of_row_and_receiver, "retained", []
    )
    retained.append((graph, x, out))
    test_rank_order_is_independent_of_row_and_receiver.retained = retained
