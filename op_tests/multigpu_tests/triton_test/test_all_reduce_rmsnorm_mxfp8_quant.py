# SPDX-License-Identifier: MIT
"""TP4 Gluon AR + Gemma RMSNorm + native group-32 MXFP8 exactness."""

import multiprocessing as mp
import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.dist.communication_op import (
    tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant,
)
from aiter.dist.parallel_state import get_tp_group, set_custom_all_reduce
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.ops.quant import per_1x32_mx_quant_hip
from aiter.ops.triton.comms.all_reduce import (
    all_reduce_free,
    all_reduce_init,
    all_reduce_rmsnorm_mxfp8_quant,
    all_reduce_rmsnorm_per_token_quant,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import ensure_spawn_method

WORLD_SIZE, HIDDEN, EPS = 4, 6144, 1e-6
TOKENS = (0, 1, 8, 16, 32, 64, 96, 120, 128, 240, 256)
SKIP = torch.cuda.device_count() < WORLD_SIZE or arch_info.get_arch() != "gfx1250"
pytestmark = pytest.mark.skipif(SKIP, reason="needs 4 gfx1250 GPUs")


def _inputs(tokens, rank, seed):
    device = torch.device(f"cuda:{rank}")
    generator = torch.Generator(device=device).manual_seed(seed)
    inputs = [
        torch.randn(
            (tokens, HIDDEN),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(WORLD_SIZE)
    ]
    residual = torch.randn(
        (tokens, HIDDEN),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    weight = (
        torch.randn(
            HIDDEN,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.1
    )
    if tokens:
        for tensor in inputs:
            tensor[0].zero_()
            tensor[-1].mul_(1e-8)
        residual[0].zero_()
        residual[-1].mul_(1e-8)
    return inputs, residual, weight


def _split(x, residual, weight, group, inboxes, calls):
    normed, residual_out = all_reduce_rmsnorm_per_token_quant(
        x,
        residual,
        weight,
        EPS,
        True,
        False,
        group,
        inboxes,
        calls,
    )
    values, scales = per_1x32_mx_quant_hip(
        normed,
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
    )
    return values, residual_out, scales


def _same_bits(got, want):
    return all(
        torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        )
        for actual, expected in zip(got, want)
    )


def _same_ranks(outputs, group):
    for output in outputs:
        raw = output.contiguous().view(torch.uint8)
        peers = [torch.empty_like(raw) for _ in range(WORLD_SIZE)]
        dist.all_gather(peers, raw, group=group)
        if any(not torch.equal(peer, raw) for peer in peers):
            return False
    return True


def _worker(rank, init_method):
    torch.cuda.set_device(rank)
    set_custom_all_reduce(False)
    os.environ["AITER_GLUON_FUSED_AR"] = "1"
    aiter.init_dist_env(WORLD_SIZE, rank, distributed_init_method=init_method)
    tp = get_tp_group()
    group = tp.cpu_group
    inboxes, calls = all_reduce_init(group, 4 << 20)
    failures = []
    try:
        for tokens in TOKENS:
            inputs, residual, weight = _inputs(tokens, rank, 100 + tokens)
            got = all_reduce_rmsnorm_mxfp8_quant(
                inputs[rank],
                residual,
                weight,
                EPS,
                True,
                group,
                inboxes,
                calls,
            )
            if tokens:
                want = _split(
                    inputs[rank],
                    residual,
                    weight,
                    group,
                    inboxes,
                    calls,
                )
                if not _same_bits(got, want):
                    failures.append(f"rank {rank}: eager M={tokens} differs")
                if not _same_ranks(got, tp.device_group):
                    failures.append(f"rank {rank}: M={tokens} ranks differ")
            elif any(output.numel() for output in got):
                failures.append(f"rank {rank}: zero-row output is not empty")

            with patch(
                "aiter.ops.triton.comms.all_reduce." "all_reduce_rmsnorm_mxfp8_quant",
                wraps=all_reduce_rmsnorm_mxfp8_quant,
            ) as launch:
                public = tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant(
                    inputs[rank],
                    residual,
                    weight,
                    EPS,
                    gemma_norm=True,
                )
            if launch.called != bool(tokens):
                failures.append(f"rank {rank}: M={tokens} dispatch differs")
            if tokens and not _same_bits(public, want):
                failures.append(f"rank {rank}: public M={tokens} differs")

        if calls[-1].item():
            failures.append(f"rank {rank}: poll timeout")
        return failures
    finally:
        all_reduce_free(group, inboxes)
        aiter.destroy_dist_env()


def test_all_reduce_rmsnorm_mxfp8_quant():
    ensure_spawn_method()
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with mp.Pool(processes=WORLD_SIZE) as pool:
        results = [
            pool.apply_async(_worker, (rank, init_method)) for rank in range(WORLD_SIZE)
        ]
        failures = [
            failure for result in results for failure in result.get(timeout=1800)
        ]
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    test_all_reduce_rmsnorm_mxfp8_quant()
