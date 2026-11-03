# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP4 validation for the gfx1250 Gluon AR + Gemma RMSNorm (+ FP8) path.

This is intentionally a manual GPU script, not a CPU-CI test. Run it only on four
available gfx1250 GPUs:

    AITER_GLUON_FUSED_AR=1 \
    AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
    AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=1 \
    torchrun --standalone --nproc-per-node=4 \
      op_tests/multigpu_tests/validate_gluon_fused_ar_rmsnorm_6144.py

It checks M=1,4,16,64, eager and HIP graph replay, per-token FP8 after
dequantization, rank agreement, explicit split fallbacks, and 1-vs-2/3 kernel
counts. It does not create or manage containers.
"""

import argparse
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

WORLD_SIZE = 4
HIDDEN = 6144
TOKENS = (1, 4, 16, 64)
EPS = 1e-6


@contextmanager
def _force_split_fallback(communicator):
    """Disable fused transports without destroying their owned resources."""
    saved = {
        name: getattr(communicator, name)
        for name in ("gluon_ar", "ca_comm", "qr_comm")
    }
    communicator.gluon_ar = None
    communicator.ca_comm = None
    communicator.qr_comm = None
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(communicator, name, value)


def _inputs(tokens, rank, device):
    gen = torch.Generator(device=device).manual_seed(393_000 + tokens)
    shape = (tokens, HIDDEN)
    xs = [
        torch.randn(shape, generator=gen, dtype=torch.bfloat16, device=device)
        * 0.25
        for _ in range(WORLD_SIZE)
    ]
    residual = torch.randn(
        shape, generator=gen, dtype=torch.bfloat16, device=device
    )
    weight = (
        torch.randn(HIDDEN, generator=gen, dtype=torch.bfloat16, device=device)
        * 0.1
    )
    summed = xs[0].float() + xs[1].float() + xs[2].float() + xs[3].float()
    ref_residual_fp32 = summed + residual.float()
    ref_norm_fp32 = (
        ref_residual_fp32
        * torch.rsqrt(ref_residual_fp32.square().mean(-1, keepdim=True) + EPS)
        * (1.0 + weight.float())
    )
    return (
        xs[rank],
        residual,
        weight,
        ref_residual_fp32,
        ref_norm_fp32,
    )


def _assert_rank_equal(tensor, group, name):
    bits = tensor.contiguous().view(torch.uint8)
    peers = [torch.empty_like(bits) for _ in range(WORLD_SIZE)]
    dist.all_gather(peers, bits, group=group)
    assert all(torch.equal(bits, peer) for peer in peers), (
        f"{name}: ranks produced different bits"
    )


def _assert_graph_equal(actual, expected, name):
    assert len(actual) == len(expected)
    for index, (got, want) in enumerate(zip(actual, expected)):
        assert torch.equal(
            got.contiguous().view(torch.uint8),
            want.contiguous().view(torch.uint8),
        ), f"{name}[{index}]: HIP graph result differs from eager"


def _capture(fn, graph_capture, replays):
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as capture, torch.cuda.graph(graph, stream=capture.stream):
        outputs = fn()
    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize()
    return outputs, graph


def _profile_kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [
        event.name
        for event in prof.events()
        if event.device_type.name == "CUDA"
    ]


def _validate_case(
    tokens,
    rank,
    device,
    tp,
    quantize,
    dtypes,
    graph_capture,
    tensor_model_parallel_fused_allreduce_rmsnorm,
    tensor_model_parallel_fused_allreduce_rmsnorm_quant,
    graph_replays,
):
    x, residual, weight, ref_residual, ref_norm = _inputs(tokens, rank, device)

    norm, res_out = tensor_model_parallel_fused_allreduce_rmsnorm(
        x, residual, weight, EPS, gemma_norm=True
    )
    assert torch.equal(res_out, ref_residual.to(torch.bfloat16))
    torch.testing.assert_close(
        norm.float(), ref_norm.to(torch.bfloat16).float(), rtol=2e-2, atol=2e-2
    )
    _assert_rank_equal(norm, tp.device_group, f"M={tokens} norm")
    _assert_rank_equal(res_out, tp.device_group, f"M={tokens} residual")

    q, quant_res, scale = tensor_model_parallel_fused_allreduce_rmsnorm_quant(
        x,
        residual,
        weight,
        EPS,
        quant_type="per_token",
        gemma_norm=True,
    )
    assert q.dtype == dtypes.fp8
    assert scale.dtype == torch.float32
    assert scale.shape == (tokens, 1)
    assert torch.equal(quant_res, res_out)
    q_ref, scale_ref = quantize(norm, quant_dtype=dtypes.fp8)
    dequant = q.float() * scale
    dequant_ref = q_ref.float() * scale_ref
    torch.testing.assert_close(dequant, dequant_ref, rtol=5e-2, atol=5e-2)
    _assert_rank_equal(q, tp.device_group, f"M={tokens} fp8")
    _assert_rank_equal(scale, tp.device_group, f"M={tokens} scale")

    if graph_replays:
        graph_norm, graph0 = _capture(
            lambda: tensor_model_parallel_fused_allreduce_rmsnorm(
                x, residual, weight, EPS, gemma_norm=True
            ),
            graph_capture,
            graph_replays,
        )
        graph_quant, graph1 = _capture(
            lambda: tensor_model_parallel_fused_allreduce_rmsnorm_quant(
                x,
                residual,
                weight,
                EPS,
                quant_type="per_token",
                gemma_norm=True,
            ),
            graph_capture,
            graph_replays,
        )
        _assert_graph_equal(graph_norm, (norm, res_out), f"M={tokens} norm")
        _assert_graph_equal(
            graph_quant, (q, quant_res, scale), f"M={tokens} quant"
        )
        # Keep graph-owned allocations alive through the comparisons above.
        del graph0, graph1

    tp.barrier()
    with _force_split_fallback(tp.device_communicator):
        fallback_norm, fallback_res = (
            tensor_model_parallel_fused_allreduce_rmsnorm(
                x, residual, weight, EPS, gemma_norm=True
            )
        )
        fallback_q, fallback_q_res, fallback_scale = (
            tensor_model_parallel_fused_allreduce_rmsnorm_quant(
                x,
                residual,
                weight,
                EPS,
                quant_type="per_token",
                gemma_norm=True,
            )
        )
    tp.barrier()
    torch.testing.assert_close(
        fallback_res.float(), res_out.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        fallback_norm.float(), norm.float(), rtol=3e-2, atol=5e-2
    )
    torch.testing.assert_close(
        fallback_q.float() * fallback_scale,
        dequant,
        rtol=7e-2,
        atol=7e-2,
    )
    torch.testing.assert_close(
        fallback_q_res.float(), quant_res.float(), rtol=2e-2, atol=2e-2
    )

    return x, residual, weight


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-replays",
        type=int,
        default=10,
        help="HIP graph replays per shape; 0 disables graph validation",
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip the eager 1-vs-2/3 kernel-count check",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE:
        raise RuntimeError("launch with torchrun --standalone --nproc-per-node=4")

    # These must be set before importing aiter and constructing the communicator.
    os.environ.setdefault("AITER_GLUON_FUSED_AR", "1")
    os.environ.setdefault("AITER_QUICK_REDUCE_QUANTIZATION", "INT4")
    os.environ.setdefault("AITER_QUICK_REDUCE_CAST_BF16_TO_FP16", "1")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    import aiter
    from aiter import dtypes, get_hip_quant
    from aiter.dist.communication_op import (
        tensor_model_parallel_fused_allreduce_rmsnorm,
        tensor_model_parallel_fused_allreduce_rmsnorm_quant,
    )
    from aiter.dist.parallel_state import get_tp_group, graph_capture
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx1250":
        raise RuntimeError("this validation requires gfx1250")

    aiter.init_dist_env(
        WORLD_SIZE,
        rank,
        distributed_init_method="env://",
        local_rank=local_rank,
    )
    tp = get_tp_group()
    quantize = get_hip_quant(aiter.QuantType.per_Token)
    try:
        assert tp.device_communicator.gluon_ar is not None, (
            "Gluon fused AR was not initialized"
        )
        retained = {}
        for tokens in TOKENS:
            retained[tokens] = _validate_case(
                tokens,
                rank,
                device,
                tp,
                quantize,
                dtypes,
                graph_capture,
                tensor_model_parallel_fused_allreduce_rmsnorm,
                tensor_model_parallel_fused_allreduce_rmsnorm_quant,
                args.graph_replays,
            )

        if not args.skip_profile:
            x, residual, weight = retained[16]
            fused_norm = _profile_kernel_names(
                lambda: tensor_model_parallel_fused_allreduce_rmsnorm(
                    x, residual, weight, EPS, gemma_norm=True
                )
            )
            fused_quant = _profile_kernel_names(
                lambda: tensor_model_parallel_fused_allreduce_rmsnorm_quant(
                    x,
                    residual,
                    weight,
                    EPS,
                    quant_type="per_token",
                    gemma_norm=True,
                )
            )
            tp.barrier()
            with _force_split_fallback(tp.device_communicator):
                fallback_norm = _profile_kernel_names(
                    lambda: tensor_model_parallel_fused_allreduce_rmsnorm(
                        x, residual, weight, EPS, gemma_norm=True
                    )
                )
                fallback_quant = _profile_kernel_names(
                    lambda: tensor_model_parallel_fused_allreduce_rmsnorm_quant(
                        x,
                        residual,
                        weight,
                        EPS,
                        quant_type="per_token",
                        gemma_norm=True,
                    )
                )
            tp.barrier()
            assert len(fused_norm) == 1, fused_norm
            assert len(fused_quant) == 1, fused_quant
            assert "_all_reduce_rmsnorm_kernel" in fused_norm[0], fused_norm
            assert "_all_reduce_rmsnorm_kernel" in fused_quant[0], fused_quant
            assert len(fallback_norm) == 2, fallback_norm
            assert len(fallback_quant) == 3, fallback_quant
            if rank == 0:
                print(
                    "kernel counts:",
                    {
                        "norm": (len(fallback_norm), len(fused_norm)),
                        "quant": (len(fallback_quant), len(fused_quant)),
                    },
                )
        poll_timeouts = tp.device_communicator.gluon_ar[1][-1].item()
        assert poll_timeouts == 0, (
            f"Gluon receiver polls timed out {poll_timeouts} times"
        )
        if rank == 0:
            print("PASS: TP4 hidden=6144 Gemma fused AR validation")
    finally:
        torch.cuda.synchronize()
        tp.barrier()
        aiter.destroy_dist_env()


if __name__ == "__main__":
    main()
