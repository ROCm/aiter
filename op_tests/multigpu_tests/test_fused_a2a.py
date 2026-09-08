# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Multi-rank correctness for combined/split Ulysses in-hop and world-8 out-hop."""

from __future__ import annotations

import os
import socket

import mori.shmem as ms
import pytest
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


def _norm_rope(input_tensor, weight, cos, sin, dtype=torch.bfloat16):
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
    return output.to(dtype)


def _mx_fp8_reference(values):
    """E4M3 RNE with RoundUp E8M0 scaling, independently expressed in torch."""
    blocks = values.float().reshape(*values.shape[:-1], -1, 32)
    amax = blocks.abs().amax(dim=-1)
    inv_max = torch.tensor(0x3B124925, dtype=torch.int32, device=values.device).view(
        torch.float32
    )
    bits = (amax * inv_max).view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).int()
    exponent = exponent.clamp(0, 255)
    reciprocal = ((254 - exponent) << 23).view(torch.float32)
    payload = (blocks * reciprocal.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return payload.view(torch.uint8).reshape_as(values), exponent.to(torch.uint8)


def _mx_int8_reference(values):
    """Symmetric RNE INT8 with the golden block-32 E8M0 scale rule."""
    blocks = values.float().reshape(*values.shape[:-1], -1, 32)
    need = (blocks.abs().amax(dim=-1) / 127.0).clamp_min(1.0e-30)
    exponent = (torch.ceil(torch.log2(need)) + 127.0).clamp(0, 254)
    scale = torch.exp2(exponent - 127.0)
    payload = torch.round(blocks / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return payload.view(torch.uint8).reshape_as(values), exponent.to(torch.uint8)


def _dequantize(payload, scales, dtype=torch.float32, codec="e4m3"):
    value_dtype = torch.int8 if codec == "int8" else torch.float8_e4m3fn
    blocks = payload.view(value_dtype).float().reshape(*scales.shape, 32)
    scale = torch.exp2(scales.float() - 127)
    return (blocks * scale.unsqueeze(-1)).reshape_as(payload).to(dtype)


def _assert_quantized(
    payload, scales, reference, reference_scales, oracle, label, codec="e4m3"
):
    if payload.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise AssertionError(f"{label}: payload and E8M0 scales must be uint8")
    _assert_equal(scales, reference_scales, f"{label} scales")
    actual = _dequantize(payload, scales, codec=codec)
    return _assert_quantized_values(actual, reference, oracle, label)


def _assert_quantized_values(actual, reference, oracle, label):
    format_sqnr, _, _ = _metrics(actual, reference)
    mismatch_fraction = (actual != reference).float().mean().item()
    # FP32 norm/RoPE can cross a codec rounding midpoint, but only very rarely.
    if not (format_sqnr >= 60.0 and mismatch_fraction <= 1.0e-4):
        raise AssertionError(
            f"{label}: format SQNR={format_sqnr:.4f} dB, "
            f"mismatch_fraction={mismatch_fraction:.6g}"
        )
    reference_sqnr, _, _ = _metrics(reference, oracle)
    kernel_sqnr, _, _ = _metrics(actual, oracle)
    if not kernel_sqnr >= reference_sqnr - 2.0:
        raise AssertionError(
            f"{label}: kernel SQNR={kernel_sqnr:.4f} dB, "
            f"reference SQNR={reference_sqnr:.4f} dB (margin=2 dB)"
        )
    return (
        f"{label}: format-correctness=PASS format-SQNR={format_sqnr:.4f}dB "
        f"mismatch_fraction={mismatch_fraction:.6g} "
        f"reference-SQNR={reference_sqnr:.4f}dB kernel-SQNR={kernel_sqnr:.4f}dB"
    )


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
    codec = os.environ.get("FUSED_A2A_CODEC", "e4m3")
    quant_reference = _mx_int8_reference if codec == "int8" else _mx_fp8_reference
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
            if case_name == "small":
                q[:, 0].zero_()
                q[:, 1] = (
                    torch.arange(heads * head_dim, device=device) % 127 - 63
                ).reshape(heads, head_dim).float() / 32
            k = q * 0.75 + 0.125
            v = q + 2
            if case_name == "small":
                v[:, 0].zero_()
                # Distinct block scales and isolated outliers in the final peer group.
                blocks = v[:, -1].view(heads, head_dim // 32, 32)
                block_id = torch.arange(heads * (head_dim // 32), device=device)
                amplitude = torch.exp2((block_id % 9 - 4).float()).view(heads, -1, 1)
                pattern = (torch.arange(32, device=device).float() - 15) / 17
                blocks.copy_(pattern * amplitude)
                blocks[..., 31] = (63.75 * amplitude.squeeze(-1)).to(torch.bfloat16)
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

            for fuse_norm_rope in ((True, False) if case_name == "small" else (True,)):
                quant_inputs = (
                    (
                        _norm_rope(q, norm_q, cos, sin, torch.float32),
                        _norm_rope(k, norm_k, cos, sin, torch.float32),
                        v,
                    )
                    if fuse_norm_rope
                    else inputs
                )
                quantized = [quant_reference(input) for input in quant_inputs]
                quant_references = a2a_references(
                    [
                        _dequantize(payload, scale, codec=codec)
                        for payload, scale in quantized
                    ],
                    heads_local,
                    seq_len,
                    head_dim,
                )
                scale_references = a2a_references(
                    [scale for _, scale in quantized],
                    heads_local,
                    seq_len,
                    head_dim // 32,
                )
                expected = references if fuse_norm_rope else transport_references
                aux = (norm_q, norm_k, cos, sin) if fuse_norm_rope else ()
                modes = (
                    ((False, "bf16"), (False, "fp8"), (True, "bf16"), (True, "fp8"))
                    if fuse_norm_rope
                    else ((False, "bf16"), (True, "fp8"))
                )
                for split, return_mode in modes:
                    op = FusedA2AIntraNodeOp(
                        rank=rank,
                        world_size=world_size,
                        shape=q.shape,
                        fuse_norm_rope=fuse_norm_rope,
                        split=split,
                        quant=True,
                        return_mode=None if return_mode == "bf16" else return_mode,
                    )
                    for epoch in range(3):
                        result = op(*inputs, *aux)
                        torch.cuda.synchronize()
                        quant_metrics = []
                        if return_mode == "fp8":
                            actuals, scales = result
                            for i, tensor_name in enumerate("qkv"):
                                quant_metrics.append(
                                    _assert_quantized(
                                        actuals[i].view(expected_shape),
                                        scales[i].view(scale_references[i].shape),
                                        quant_references[i],
                                        scale_references[i],
                                        expected[i],
                                        tensor_name,
                                        codec=codec,
                                    )
                                )
                        else:
                            actuals = result
                            for i, tensor_name in enumerate("qkv"):
                                actual = actuals[i].view(expected_shape)
                                if actual.dtype != torch.bfloat16:
                                    raise AssertionError(
                                        "bf16 return must produce bf16 Q/K/V"
                                    )
                                parity = epoch % 2
                                received_reference = _dequantize(
                                    op.outputs_sets[parity][i].view(expected_shape),
                                    op.scales_sets[parity][i].view(
                                        scale_references[i].shape
                                    ),
                                    torch.bfloat16,
                                    codec=codec,
                                )
                                _assert_equal(actual, received_reference, tensor_name)
                                quant_metrics.append(
                                    _assert_quantized_values(
                                        actual,
                                        quant_references[i].to(torch.bfloat16),
                                        expected[i],
                                        tensor_name,
                                    )
                                    + " native-dequant=bit-exact"
                                )
                        if case_name == "small":
                            actual_v = actuals[2].view(expected_shape)
                            if return_mode == "fp8":
                                actual_v = _dequantize(
                                    actual_v,
                                    scales[2].view(scale_references[2].shape),
                                    codec=codec,
                                )
                            # Outlier absmax coarsens uniform INT8's bulk; deployed V lacks this pattern and gains 11.6 dB.
                            min_v_sqnr = 31.0 if codec == "int8" else 40.0
                            v_sqnr, _, _ = _metrics(actual_v, expected[2])
                            if not v_sqnr >= min_v_sqnr:
                                raise AssertionError(
                                    f"small {codec} V: SQNR={v_sqnr:.4f} dB < {min_v_sqnr} dB"
                                )
                        dist.barrier()
                    if rank == 0:
                        print(
                            f"PASS {case_name} quant=True codec={codec} split={split} "
                            f"norm_rope={fuse_norm_rope} return={return_mode}: "
                            + " ".join(quant_metrics)
                            + " epochs=3",
                            flush=True,
                        )

            if world_size >= 2:
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
                    f"SKIP: {case_name} out-hop requires world_size>=2",
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
    passed = sum(
        8 + 2 * (name == "small") + int(_WORLD_SIZE >= 2) for name, *_ in _CASES
    )
    skipped = len(_CASES) * int(_WORLD_SIZE < 2)
    print(f"{passed} passed, {skipped} skipped on {arch}")
    return 0


def test_fused_a2a(capfd):
    if not torch.cuda.is_available():
        pytest.skip("fused_a2a requires ROCm GPUs")
    arch = get_gfx_runtime()
    if arch != "gfx950":
        pytest.skip(f"fused_a2a supports gfx950, attached GPU is {arch}")
    if torch.cuda.device_count() < _WORLD_SIZE:
        pytest.skip(f"fused_a2a requires {_WORLD_SIZE} visible GPUs")
    # Keep per-hop codec quality visible with the standard pytest -rs invocation.
    with capfd.disabled():
        assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
