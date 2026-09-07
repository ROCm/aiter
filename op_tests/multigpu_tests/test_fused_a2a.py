# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Multi-rank correctness for combined/split Ulysses in-hop and world-8 out-hop."""

from __future__ import annotations

import os
import socket

import mori.shmem as ms
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.multiprocessing as mp

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.fused_a2a_intranode_op import (
    FusedA2AIntraNodeOp,
    FusedA2AOutIntraNodeOp,
)

_WORLD_SIZE = int(os.environ.get("FUSED_A2A_WORLD_SIZE", "8"))
_CASES = (
    ("small", 8, 17, 128),
    ("deployed", 40, 9419, 128),
)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _sequence_major_input(rank, heads, seq_len, head_dim, device):
    generator = torch.Generator(device="cpu").manual_seed(1103 + rank)
    return (
        torch.randn(
            (1, seq_len, heads, head_dim), generator=generator, dtype=torch.float32
        )
        .to(torch.bfloat16)
        .to(device)
    )


def _norm_rope(input_tensor, weight, cos, sin):
    heads, head_dim = input_tensor.shape[-2:]
    norm = torch.nn.RMSNorm(
        heads * head_dim,
        eps=1.0e-6,
        elementwise_affine=True,
        device=input_tensor.device,
        dtype=torch.float32,
    )
    norm.weight.data.copy_(weight.float())
    values = norm(input_tensor.flatten(-2).float()).view_as(input_tensor.float())
    even = values[..., 0::2]
    odd = values[..., 1::2]
    output = torch.empty_like(values)
    output[..., 0::2] = even * cos[..., 0::2] - odd * sin[..., 1::2]
    output[..., 1::2] = even * sin[..., 1::2] + odd * cos[..., 0::2]
    return output.to(torch.bfloat16)


def _metrics(actual, reference):
    actual = actual.float()
    reference = reference.float()
    error = actual - reference
    sqnr = 10.0 * torch.log10(reference.square().sum() / error.square().sum())
    rel_mae = error.abs().mean() / reference.abs().mean()
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), reference.flatten(), dim=0
    )
    return sqnr.item(), rel_mae.item(), cosine.item()


def _assert_equal(actual, reference, label):
    if not torch.equal(actual, reference):
        mismatch = torch.nonzero(actual != reference, as_tuple=False)[0].flatten()
        raise AssertionError(
            f"{label}: byte mismatch at index {mismatch.tolist()}: "
            f"actual={actual[tuple(mismatch)].item()} "
            f"reference={reference[tuple(mismatch)].item()}"
        )


def _run_rank(rank, world_size, port):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    try:
        cpu_group = dist.new_group(backend="gloo")
        torch._C._distributed_c10d._register_process_group("mori", cpu_group)
        ms.shmem_torch_process_group_init("mori")

        for case_name, heads, seq_len, head_dim in _CASES:
            q = _sequence_major_input(rank, heads, seq_len, head_dim, device)
            k = q * 0.75 + 0.125
            v = q + 2
            hd = heads * head_dim
            norm_q = torch.linspace(0.5, 1.5, hd, device=device).to(torch.bfloat16)
            norm_k = torch.linspace(1.5, 0.5, hd, device=device).to(torch.bfloat16)
            angles = (
                torch.arange(seq_len, device=device, dtype=torch.float32).view(
                    1, seq_len, 1, 1
                )
                * torch.arange(head_dim // 2, device=device, dtype=torch.float32).view(
                    1, 1, 1, -1
                )
                / 10000.0
            )
            cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1).contiguous()
            sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1).contiguous()
            transformed = (
                _norm_rope(q, norm_q, cos, sin),
                _norm_rope(k, norm_k, cos, sin),
                v,
            )
            inputs = (q, k, v)
            heads_local = heads // world_size
            expected_shape = (1, heads_local, world_size * seq_len, head_dim)

            def a2a_references(tensors, heads_local, seq_len, head_dim):
                references = []
                for input_tensor in tensors:
                    packed = input_tensor.permute(0, 2, 1, 3).contiguous()
                    gathered = funcol.wait_tensor(
                        funcol.all_to_all_single(
                            packed.view(-1), None, None, dist.group.WORLD
                        )
                    )
                    reference = gathered.view(
                        world_size, 1, heads_local, seq_len, head_dim
                    )
                    references.append(
                        reference.permute(1, 2, 0, 3, 4).reshape(
                            1, heads_local, world_size * seq_len, head_dim
                        )
                    )
                return references

            references = a2a_references(transformed, heads_local, seq_len, head_dim)
            transport_references = a2a_references(
                inputs, heads_local, seq_len, head_dim
            )

            for split in (False, True):
                for fuse_norm_rope in (True, False):
                    op = FusedA2AIntraNodeOp(
                        rank=rank,
                        world_size=world_size,
                        shape=q.shape,
                        dtype=q.dtype,
                        fuse_norm_rope=fuse_norm_rope,
                        split=split,
                    )
                    expected = references if fuse_norm_rope else transport_references
                    aux = (norm_q, norm_k, cos, sin) if fuse_norm_rope else ()
                    # Cover the compiled fast path and both receive-buffer parities.
                    for epoch in range(3):
                        actuals = op(*inputs, *aux)
                        torch.cuda.synchronize()
                        case_metrics = []
                        for tensor_name, actual, reference in zip(
                            "qkv", actuals, expected, strict=True
                        ):
                            actual = actual.view(reference.shape)
                            label = (
                                f"{case_name} split={split} norm_rope={fuse_norm_rope} "
                                f"{tensor_name} rank {rank} epoch {epoch}"
                            )
                            if tuple(actual.shape) != expected_shape:
                                raise AssertionError(
                                    f"{label}: got {tuple(actual.shape)}, "
                                    f"expected {expected_shape}"
                                )
                            if tensor_name == "v" or not fuse_norm_rope:
                                _assert_equal(actual, reference, label)
                            else:
                                sqnr, rel_mae, cosine = _metrics(actual, reference)
                                if sqnr < 40.0:
                                    raise AssertionError(
                                        f"{label}: SQNR {sqnr:.2f} dB < 40 dB "
                                        f"(rel-MAE={rel_mae:.6g}, cosine={cosine:.9f})"
                                    )
                                case_metrics.append(
                                    (tensor_name, sqnr, rel_mae, cosine)
                                )
                        dist.barrier()
                    if rank == 0:
                        metric_text = " ".join(
                            f"{name}: SQNR={sqnr:.2f}dB rel-MAE={rel_mae:.6g} "
                            f"cos={cosine:.9f}"
                            for name, sqnr, rel_mae, cosine in case_metrics
                        )
                        exact = "full-V" if fuse_norm_rope else "transport-QKV"
                        print(
                            f"PASS {case_name} split={split} norm_rope={fuse_norm_rope}: "
                            f"{metric_text} {exact}=byte-identical epochs=3 "
                            f"in={tuple(q.shape)} gathered={expected_shape}",
                            flush=True,
                        )

            if world_size >= 8:
                out_input = references[0].contiguous()
                out_packed = out_input.permute(2, 0, 1, 3).contiguous()
                out_reference = funcol.wait_tensor(
                    funcol.all_to_all_single(
                        out_packed.view(-1), None, None, dist.group.WORLD
                    )
                ).view(1, seq_len, heads, head_dim)
                out_op = FusedA2AOutIntraNodeOp(
                    rank=rank,
                    world_size=world_size,
                    shape=out_input.shape,
                    dtype=out_input.dtype,
                )
                out_actual = out_op(out_input).view(out_reference.shape)
                torch.cuda.synchronize()
                _assert_equal(out_actual, out_reference, f"{case_name} out rank {rank}")
                dist.barrier()
                if rank == 0:
                    print(f"PASS {case_name} out-hop=byte-identical", flush=True)
            elif rank == 0:
                print(
                    f"SKIP: {case_name} out-hop requires world_size>=8 "
                    "(_OUT_CHANNEL_COUNT)",
                    flush=True,
                )
    finally:
        try:
            ms.shmem_finalize()
        finally:
            dist.destroy_process_group()


def main():
    if not torch.cuda.is_available():
        print("SKIP: fused_a2a requires ROCm GPUs")
        return 0
    arch = get_gfx_runtime()
    if arch != "gfx950":
        print(f"SKIP: fused_a2a M1 supports gfx950, attached GPU is {arch}")
        return 0
    if torch.cuda.device_count() < _WORLD_SIZE:
        print(
            f"SKIP: fused_a2a requires {_WORLD_SIZE} visible GPUs, "
            f"found {torch.cuda.device_count()}"
        )
        return 0

    mp.spawn(_run_rank, args=(_WORLD_SIZE, _free_port()), nprocs=_WORLD_SIZE, join=True)
    passed = len(_CASES) * (4 + int(_WORLD_SIZE >= 8))
    skipped = len(_CASES) * int(_WORLD_SIZE < 8)
    print(f"{passed} passed, {skipped} skipped on {arch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
