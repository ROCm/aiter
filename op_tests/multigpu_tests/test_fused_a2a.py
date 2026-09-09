# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Multi-rank correctness for combined/split Ulysses in-hop and out-hop."""

from __future__ import annotations

import math
import os
import socket

# The full codec sweep retains symmetric allocations until shmem_finalize.
os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "12G")

import flydsl.compiler as flyc
import flydsl.expr as fx
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

_WORLD_SIZE = int(os.environ.get("FUSED_A2A_WORLD_SIZE", "4"))
_CODEC_ROWS = (
    ("int8", ("int8", "int8", "int8")),
    ("i8fp8", ("int8", "int8", "e4m3")),
    ("fp8", ("e4m3", "e4m3", "e4m3")),
    ("mxfp4", ("mxfp4", "mxfp4", "e4m3")),
    ("f4f4", ("mxfp4", "mxfp4", "mxfp4")),
    ("mixed", ("int8", "e4m3", "mxfp4")),
    ("mxfp6", ("mxfp6", "mxfp6", "e4m3")),
    ("f6f4", ("mxfp6", "mxfp6", "mxfp4")),
)
_CASES = (
    ("small", 8, 17, 128),
    ("deployed", 40, 9419, 128),
)


def _hadamard_matrix(head_dim, device):
    matrix = torch.ones((1, 1), device=device)
    while matrix.shape[0] < head_dim:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0
        )
    return matrix


def _check_hadamard_precision():
    from flydsl.expr import T

    from aiter.ops.flydsl.kernels.buffer_ops import (
        buffer_load,
        buffer_store,
        create_buffer_resource_from_addr,
    )
    from aiter.ops.flydsl.kernels.fused_a2a_intranode_kernel import _hadamard_head

    @flyc.kernel
    def rotate(src: fx.Int64, dst: fx.Int64):
        lane = fx.Int32(fx.gpu.thread_id("x"))
        offset = fx.Int32(fx.gpu.block_id("x") * 512) + lane * 8
        source = create_buffer_resource_from_addr(src)
        output = create_buffer_resource_from_addr(dst)
        low = fx.Vector(buffer_load(source, offset, vec_width=4, dtype=T.f32))
        high = fx.Vector(buffer_load(source, offset + 4, vec_width=4, dtype=T.f32))
        values = [low[i] for i in range(4)] + [high[i] for i in range(4)]
        result = _hadamard_head(values, lane, 128)
        for half in range(2):
            part = fx.Vector.from_elements(
                [result[half * 4 + i] for i in range(4)], fx.Float32
            )
            buffer_store(part, output, offset + half * 4)

    @flyc.jit
    def launch(src: fx.Int64, dst: fx.Int64):
        rotate(src, dst).launch(grid=(8,), block=(64,))

    generator = torch.Generator().manual_seed(417)
    inputs = torch.randn(2, 16, 128, generator=generator).cuda()
    actual = torch.empty_like(inputs)
    launch(fx.Int64(inputs.data_ptr()), fx.Int64(actual.data_ptr()))
    matrix = _hadamard_matrix(128, inputs.device)
    torch.testing.assert_close(
        actual, (inputs @ matrix) * 128**-0.5, atol=2e-6, rtol=2e-6
    )
    scores = inputs[0] @ inputs[1].T
    rotated_scores = actual[0] @ actual[1].T
    torch.testing.assert_close(rotated_scores, scores, atol=2e-5, rtol=2e-6)
    relative = (rotated_scores - scores).norm() / scores.norm()
    print(
        f"PASS Hadamard FP32 score-invariance: relative-L2={relative.item():.9g} "
        f"max-abs={(rotated_scores - scores).abs().max().item():.9g}",
        flush=True,
    )


def _check_fp6_converters():
    from flydsl.expr import T, const_expr

    from aiter.ops.flydsl.kernels.buffer_ops import (
        buffer_load,
        buffer_store,
        create_buffer_resource_from_addr,
    )
    from aiter.ops.flydsl.kernels.quant_utils import emit_f32_to_e2m3, emit_f32_to_e3m2

    @flyc.kernel
    def convert(
        src: fx.Int64,
        dst: fx.Int64,
        count: fx.Constexpr[int],
        variant: fx.Constexpr[int],
    ):
        index = fx.Int32(fx.gpu.block_id("x") * 256 + fx.gpu.thread_id("x"))
        source = create_buffer_resource_from_addr(src)
        output = create_buffer_resource_from_addr(dst)
        if index < count:
            value = buffer_load(source, index, vec_width=1, dtype=T.f32)
            code = (
                emit_f32_to_e2m3(value)
                if const_expr(variant == 0)
                else emit_f32_to_e3m2(value)
            )
            buffer_store(fx.Int32(code), output, index)

    @flyc.jit
    def launch(
        src: fx.Int64,
        dst: fx.Int64,
        count: fx.Constexpr[int],
        variant: fx.Constexpr[int],
    ):
        convert(src, dst, count, variant).launch(
            grid=((count + 255) // 256,), block=(256,)
        )

    generator = torch.Generator().manual_seed(2307)
    random = torch.randn(4096, generator=generator).cuda() * 32
    for variant in ("e2m3", "e3m2"):
        levels = _fp6_levels("cuda", variant)
        midpoints = (levels[:-1] + levels[1:]) / 2
        positive = torch.cat(
            (
                levels,
                midpoints,
                torch.nextafter(midpoints, torch.full_like(midpoints, float("inf"))),
                torch.nextafter(midpoints, torch.zeros_like(midpoints)),
                torch.tensor([1.96875, 1.9375, 65504.0, 1.0e-30], device="cuda"),
            )
        )
        values = torch.cat((positive, -positive, random))
        output = torch.empty_like(values, dtype=torch.int32)
        launch(
            fx.Int64(values.data_ptr()),
            fx.Int64(output.data_ptr()),
            values.numel(),
            0 if variant == "e2m3" else 1,
        )
        torch.cuda.synchronize()
        reference = _fp6_codes(values, variant).int()
        _assert_equal(output, reference, f"{variant} converter")
        _assert_equal(
            _fp6_dequantize_codes(output, variant),
            _fp6_dequantize_codes(reference, variant),
            variant,
        )
        print(
            f"PASS {variant} converter: {values.numel()} codes, all grid/midpoint neighbors and saturation",
            flush=True,
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


def _norm_rope(
    input_tensor, weight, cos, sin, dtype=torch.bfloat16, fused_rounding=False
):
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
    if fused_rounding:
        source = input_tensor.float()
        reciprocal = torch.rsqrt(
            source.flatten(-2).square().mean(-1, keepdim=True) + 1.0e-6
        )
        values = (
            source * reciprocal.unsqueeze(-1) * weight.float().view(heads, head_dim)
        )
    even = values[..., 0::2]
    odd = values[..., 1::2]
    output = torch.empty_like(values)
    if fused_rounding:
        # Emulate the device RoPE FMA's single FP32 rounding before the BF16 boundary.
        output[..., 0::2] = (
            even.double() * cos[..., 0::2].double() - (odd * sin[..., 1::2]).double()
        )
        output[..., 1::2] = (
            even.double() * sin[..., 1::2].double() + (odd * cos[..., 0::2]).double()
        )
    else:
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


def _mx_fp4_reference(values):
    """Nearest E2M1 level, ties to even nibble, with RoundUp E8M0 scaling."""
    blocks = values.float().reshape(*values.shape[:-1], -1, 32)
    amax = blocks.abs().amax(dim=-1)
    inv_max = torch.tensor(0x3E2AAAAB, dtype=torch.int32, device=values.device).view(
        torch.float32
    )
    bits = (amax * inv_max).view(torch.int32)
    exponent = (((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).int()).clamp(0, 255)
    reciprocal = ((254 - exponent) << 23).view(torch.float32)
    scaled = blocks * reciprocal.unsqueeze(-1)
    magnitude = scaled.abs()
    nibble = torch.zeros_like(magnitude, dtype=torch.uint8)
    for i, midpoint in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)):
        above = magnitude >= midpoint if i % 2 else magnitude > midpoint
        nibble += above.to(torch.uint8)
    nibble |= torch.signbit(scaled).to(torch.uint8) << 3
    nibble = nibble.reshape(*values.shape[:-1], values.shape[-1])
    payload = nibble[..., 0::2] | (nibble[..., 1::2] << 4)
    return payload, exponent.to(torch.uint8)


def _fp6_levels(device, variant="e2m3"):
    mbits, bias = (3, 1) if variant == "e2m3" else (2, 3)
    codes = torch.arange(32, device=device)
    exponent, mantissa = codes >> mbits, codes & ((1 << mbits) - 1)
    return torch.where(
        exponent == 0,
        mantissa.float() * 2.0 ** (1 - bias - mbits),
        (1 + mantissa.float() / (1 << mbits)) * torch.exp2(exponent.float() - bias),
    )


def _fp6_codes(values, variant="e2m3"):
    levels = _fp6_levels(values.device, variant)
    magnitude = values.abs()
    code = torch.zeros_like(values, dtype=torch.uint8)
    for i in range(31):
        midpoint = (levels[i] + levels[i + 1]) / 2
        above = magnitude >= midpoint if i % 2 else magnitude > midpoint
        code += above.to(torch.uint8)
    return code | (torch.signbit(values).to(torch.uint8) << 5)


def _fp6_dequantize_codes(codes, variant="e2m3"):
    levels = _fp6_levels(codes.device, variant)
    magnitude = levels[(codes & 31).long()]
    return torch.where(codes & 32 != 0, -magnitude, magnitude)


def _mx_fp6_reference(values):
    """OCP E2M3 midpoint RNE, packed four codes per three bytes."""
    blocks = values.float().reshape(*values.shape[:-1], -1, 32)
    bits = (blocks.abs().amax(dim=-1) * (1.0 / 7.5)).view(torch.int32)
    exponent = (((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).int()).clamp(0, 255)
    reciprocal = ((254 - exponent) << 23).view(torch.float32)
    codes = _fp6_codes(blocks * reciprocal.unsqueeze(-1)).reshape(-1, 4).int()
    words = codes[:, 0] | (codes[:, 1] << 6) | (codes[:, 2] << 12) | (codes[:, 3] << 18)
    payload = torch.stack((words & 255, (words >> 8) & 255, words >> 16), dim=-1)
    return payload.to(torch.uint8).reshape(
        *values.shape[:-1], values.shape[-1] * 3 // 4
    ), exponent.to(torch.uint8)


def _dequantize(payload, scales, dtype=torch.float32, codec="e4m3"):
    if codec == "mxfp6":
        triplets = payload.reshape(-1, 3).int()
        words = triplets[:, 0] | (triplets[:, 1] << 8) | (triplets[:, 2] << 16)
        codes = torch.stack([(words >> (i * 6)) & 63 for i in range(4)], dim=-1)
        values = _fp6_dequantize_codes(codes)
    elif codec == "mxfp4":
        nibble = torch.stack((payload & 15, payload >> 4), dim=-1).long()
        levels = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=payload.device
        )
        values = levels[nibble & 7] * torch.where(nibble & 8 != 0, -1.0, 1.0)
    else:
        value_dtype = torch.int8 if codec == "int8" else torch.float8_e4m3fn
        values = payload.view(value_dtype).float()
    blocks = values.reshape(*scales.shape, 32)
    scale = torch.exp2(scales.float() - 127)
    return (blocks * scale.unsqueeze(-1)).flatten(-2).to(dtype)


def _assert_quantized(
    payload, scales, reference, reference_scales, oracle, label, codec="e4m3"
):
    if payload.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise AssertionError(f"{label}: payload and E8M0 scales must be uint8")
    _assert_equal(scales, reference_scales, f"{label} scales")
    actual = _dequantize(payload, scales, codec=codec)
    return _assert_quantized_values(actual, reference, oracle, label)


def _assert_fused_fp6_scales(actual, reference, reference_amax, label, max_value=7.5):
    different = actual != reference
    delta = (actual.int() - reference.int()).abs()
    lower = torch.minimum(actual, reference).float()
    threshold = max_value * torch.exp2(lower - 127.0)
    relative_distance = (reference_amax - threshold).abs() / threshold
    # E8M0 steps at amax / 7.5 powers of two. Different FP32 reduction orders
    # can straddle that threshold; neither ordering is intrinsically more correct.
    tie = (
        (delta == 1)
        & torch.isfinite(relative_distance)
        & (relative_distance <= 8 * torch.finfo(torch.float32).eps)
    )
    if torch.any(different & ~tie):
        _assert_equal(actual[~tie], reference[~tie], f"{label} non-tie scales")
    count = int(different.sum().item())
    fraction = count / actual.numel()
    if fraction > 1.0e-5:
        raise AssertionError(
            f"{label}: scale ties {count}/{actual.numel()} exceed 1e-5"
        )
    return count, actual.numel()


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


def _check_hadamard_transport(rank, world_size, device, a2a_references):
    heads, seq_len, head_dim = 8, 17, 128
    heads_local = heads // world_size
    shape = (1, heads_local, world_size * seq_len, head_dim)
    scale_shape = (*shape[:-1], head_dim // 32)
    matrix = _hadamard_matrix(head_dim, device)
    quantizers = {
        "int8": _mx_int8_reference,
        "e4m3": _mx_fp8_reference,
        "mxfp4": _mx_fp4_reference,
        "mxfp6": _mx_fp6_reference,
    }
    # Independent Q/K expose sign, normalization, and head-boundary errors.
    inputs = [
        _sequence_major_input(rank + role * 19, heads, seq_len, head_dim, device)
        for role in range(3)
    ]
    outliers = [value.clone() for value in inputs]
    for value in outliers[:2]:
        value[..., 13] = 90.5
    for codec in ("int8", "mxfp4", "mxfp6"):
        for role in "QKV":
            os.environ[f"FUSED_A2A_CODEC_{role}"] = codec if role != "V" else "e4m3"
        for split in (True, False):
            ops = []
            for enabled in (False, True):
                os.environ["FUSED_A2A_HADAMARD"] = str(int(enabled))
                ops.append(
                    FusedA2AIntraNodeOp(
                        rank=rank,
                        world_size=world_size,
                        shape=inputs[0].shape,
                        fuse_norm_rope=False,
                        split=split,
                        quant=True,
                        return_mode="fp8" if split else "bf16",
                    )
                )
            for epoch, source in enumerate((inputs, outliers, inputs)):
                references = a2a_references(source, heads_local, seq_len, head_dim)
                rotated_refs = [
                    (value.float() @ matrix) * head_dim**-0.5
                    for value in references[:2]
                ]
                decoded = []
                for op in ops:
                    result = op(*source)
                    parity = epoch % 2
                    payloads = op.outputs_sets[parity]
                    scales = op.scales_sets[parity]
                    values = [
                        _dequantize(payloads[i], scales[i].view(scale_shape), codec=c)
                        for i, c in enumerate((codec, codec, "e4m3"))
                    ]
                    if not split:
                        for i in range(3):
                            _assert_equal(
                                result[i].view(shape),
                                values[i].to(torch.bfloat16),
                                "Hadamard native dequant",
                            )
                    decoded.append(values)
                for kind, buffers in (
                    ("payload", [op.outputs_sets[epoch % 2][2] for op in ops]),
                    ("scales", [op.scales_sets[epoch % 2][2] for op in ops]),
                ):
                    _assert_equal(*buffers, f"Hadamard V {kind}")
                quantized_refs = []
                for i in range(2):
                    payload, scales = quantizers[codec](rotated_refs[i])
                    oracle = _dequantize(payload, scales, codec=codec)
                    quantized_refs.append(oracle)
                    _assert_quantized_values(
                        decoded[1][i], oracle, rotated_refs[i], f"Hadamard {codec} QK"
                    )
                scores = references[0].float() @ references[1].float().transpose(-1, -2)
                actual_scores = decoded[1][0] @ decoded[1][1].transpose(-1, -2)
                oracle_scores = quantized_refs[0] @ quantized_refs[1].transpose(-1, -2)
                _assert_quantized_values(
                    actual_scores, oracle_scores, scores, f"Hadamard {codec} scores"
                )
                score_sqnr = _metrics(actual_scores, scores)[0]
                if epoch == 1:
                    off_sqnr = _metrics(
                        torch.stack(decoded[0][:2]), torch.stack(references[:2])
                    )[0]
                    on_sqnr = _metrics(
                        torch.stack(decoded[1][:2]), torch.stack(rotated_refs)
                    )[0]
                    # hd128 rotation spreads the spike across all block-32 scales.
                    # INT8 benefits here, but MX scaling already localizes outlier
                    # damage: E2M1 can regress (21.27 -> 20.37 dB for channel 13
                    # set to 90.5). Quality is format/input-dependent, not a gate.
                    # MHA V4's packed API requires this Q/K preprocessing anyway.
                    if rank == 0:
                        print(
                            f"PASS Hadamard {codec} split={split} outlier-SQNR "
                            f"off={off_sqnr:.4f}dB on={on_sqnr:.4f}dB V=byte-identical",
                            flush=True,
                        )
                elif epoch == 2 and rank == 0:
                    print(
                        f"PASS Hadamard {codec} split={split} random score-SQNR="
                        f"{score_sqnr:.4f}dB epochs=3",
                        flush=True,
                    )
                dist.barrier()
    os.environ.pop("FUSED_A2A_HADAMARD", None)
    for role in "QKV":
        os.environ.pop(f"FUSED_A2A_CODEC_{role}", None)


def _check_v4_fp6_k_bytes(actual, expected, scales, label):
    _, sequence, heads, _ = scales.shape
    tiles = (sequence + 127) // 128
    device = actual.device
    s = torch.arange(sequence, device=device).view(-1, 1, 1, 1)
    h = torch.arange(heads, device=device).view(1, -1, 1, 1)
    g = torch.arange(4, device=device).view(1, 1, -1, 1)
    d = torch.arange(24, device=device).view(1, 1, 1, -1)
    base = (h * tiles + s // 128) * 17408
    offsets = base + torch.where(
        d < 16,
        s % 128 // 32 * 2048 + g * 512 + s % 32 * 16 + d,
        8192 + s % 128 // 32 * 1024 + g * 256 + s % 32 * 8 + d - 16,
    )
    valid = torch.zeros_like(actual, dtype=torch.bool)
    valid[offsets.flatten()] = True
    _assert_equal(actual[offsets], expected[offsets], f"{label} payload")
    slot = s % 32 // 16 * 256 + (s % 16 * 4 + s % 128 // 32) * 4
    a = (base + 16384 + slot + g).squeeze(-1)
    b = a + 512
    logical = scales[0]
    shifted = torch.zeros_like(logical)
    shifted[..., :3] = logical[..., 1:]
    shifted[:-1, :, 3] = logical[1:, :, 0]
    for offsets, values, region in ((a, logical, "A"), (b, shifted, "B")):
        valid[offsets.flatten()] = True
        _assert_equal(actual[offsets], values, f"{label} tail {region} mapping")
        _assert_equal(
            actual[offsets], expected[offsets], f"{label} tail {region} packer"
        )
    _assert_equal(
        actual[~valid],
        torch.zeros_like(actual[~valid]),
        f"{label} reserved/padding/slack",
    )


def _check_v4_output(rank, world_size, device, q_codec="mxfp4", k_codec="mxfp4"):
    from aiter.ops.mha_v4 import (
        mxfp6_k_view,
        quantize_mxfp4_k,
        quantize_mxfp4_q,
        quantize_mxfp6_k,
        quantize_mxfp6_q,
    )

    # Partial tiles, exact tiles, and a rank boundary inside a second K tile.
    cases = ((17, False, False), (32, True, False), (33, False, True), (17, True, True))
    heads, head_dim = 8, 128
    local_heads = heads // world_size
    os.environ["FUSED_A2A_V4_OUTPUT"] = "1"
    os.environ["FUSED_A2A_HADAMARD"] = "0"
    for role in "QKV":
        os.environ[f"FUSED_A2A_CODEC_{role}"] = (
            q_codec if role == "Q" else k_codec if role == "K" else "e4m3"
        )
    for seq_len, split, fused in cases:
        softmax_scale = 0.125 if split else head_dim**-0.5
        inputs = [
            _sequence_major_input(rank + 19 * role, heads, seq_len, head_dim, device)
            for role in range(3)
        ]
        # Nonrandom rows expose head/token permutations, zero scales and nibble signs.
        for role, value in enumerate(inputs):
            value[:, 0].zero_()
            value[:, 1] = (
                (torch.arange(heads * head_dim, device=device) % 31 - 15).reshape(
                    heads, head_dim
                )
                * (rank + role + 1)
                / 16
            )
        norm_q = torch.ones(heads * head_dim, dtype=torch.bfloat16, device=device)
        norm_k = norm_q.clone()
        angles = torch.arange(seq_len, device=device).view(1, -1, 1, 1) / 100
        cos = angles.cos().expand(1, seq_len, 1, head_dim).contiguous()
        sin = angles.sin().expand_as(cos).contiguous()
        transformed = [
            (
                _norm_rope(
                    value,
                    weight,
                    cos,
                    sin,
                    fused_rounding=(q_codec == "mxfp6" or k_codec == "mxfp6"),
                )
                if fused
                else value
            )
            for value, weight in zip(inputs[:2], (norm_q, norm_k))
        ]
        references = []
        for role, source in enumerate(transformed):
            gathered = [torch.empty_like(source) for _ in range(world_size)]
            dist.all_gather(gathered, source)
            full = torch.cat(gathered, dim=1)[
                :, :, rank * local_heads : (rank + 1) * local_heads
            ].contiguous()
            multiplier = softmax_scale * math.log2(math.e) if role == 0 else 1.0
            payload, scales = (
                (quantize_mxfp6_q if q_codec == "mxfp6" else quantize_mxfp4_q)(
                    full, multiplier
                )
                if role == 0
                else (quantize_mxfp6_k if k_codec == "mxfp6" else quantize_mxfp4_k)(
                    full
                )
            )
            if role == 1 and k_codec == "mxfp6":
                _, scales = mxfp6_k_view(
                    payload, scales, 1, seq_len * world_size, local_heads
                )
            rotated = (
                (full.float() @ _hadamard_matrix(128, device)) * 128**-0.5 * multiplier
            )
            amax = (
                rotated.to(torch.bfloat16)
                .float()
                .reshape(*scales.shape, 32)
                .abs()
                .amax(-1)
            )
            references.append((payload, scales, amax))
        op = FusedA2AIntraNodeOp(
            rank=rank,
            world_size=world_size,
            shape=inputs[0].shape,
            fuse_norm_rope=fused,
            split=split,
            quant=True,
            return_mode="fp8",
            softmax_scale=softmax_scale,
        )
        counts = [0, 0]
        for epoch in range(3):
            payloads, scales = op(*inputs, norm_q, norm_k, cos, sin)
            torch.cuda.synchronize()
            for role, (expected, expected_scales, amax) in enumerate(references):
                label = f"V4 {'QK'[role]} S={seq_len} split={split} fused={fused} rank={rank} epoch={epoch}"
                actual_scales = scales[role][: expected_scales.numel()].view_as(
                    expected_scales
                )
                if (q_codec if role == 0 else k_codec) == "mxfp6":
                    _assert_equal(actual_scales, expected_scales, f"{label} scales")
                    ties, total = 0, actual_scales.numel()
                else:
                    ties, total = _assert_fused_fp6_scales(
                        actual_scales, expected_scales, amax, label, max_value=6.0
                    )
                counts[0] += ties
                counts[1] += total
                same_scale = actual_scales == expected_scales
                if role == 1 and k_codec == "mxfp6":
                    assert payloads[role].numel() == expected.numel(), label
                    _check_v4_fp6_k_bytes(
                        payloads[role], expected, expected_scales, label
                    )
                    slack = scales[role][expected_scales.numel() :]
                    _assert_equal(
                        slack, torch.zeros_like(slack), f"{label} scale slack"
                    )
                    continue
                if role == 0:
                    byte_mask = same_scale.repeat_interleave(
                        24 if q_codec == "mxfp6" else 16, -1
                    ).flatten()
                else:
                    tiles = (seq_len * world_size + 127) // 128
                    padded = torch.zeros(
                        (1, local_heads, tiles * 128, 4),
                        dtype=torch.bool,
                        device=device,
                    )
                    padded[:, :, : seq_len * world_size] = same_scale.transpose(1, 2)
                    byte_mask = (
                        padded.reshape(1, local_heads, tiles, 128, 4)
                        .transpose(-1, -2)
                        .unsqueeze(-1)
                        .expand(-1, -1, -1, -1, -1, 16)
                        .reshape(-1)
                    )
                    # The HIP oracle leaves padding uninitialized; the V4 ABI requires zeros.
                    actual_rows = (
                        payloads[role]
                        .view(1, local_heads, tiles, 4, 128, 16)
                        .transpose(3, 4)
                        .reshape(1, local_heads, tiles * 128, 64)
                    )
                    padding = actual_rows[:, :, seq_len * world_size :]
                    _assert_equal(
                        padding, torch.zeros_like(padding), f"{label} padding"
                    )
                assert payloads[role].numel() == expected.numel(), label
                _assert_equal(
                    payloads[role][byte_mask], expected.flatten()[byte_mask], label
                )
            dist.barrier()
        totals = torch.tensor(counts, dtype=torch.int64, device=device)
        dist.all_reduce(totals)
        if rank == 0:
            print(
                f"PASS V4 Q={q_codec} K={k_codec} S={seq_len} split={split} fused={fused} epochs=3 scale-ties={totals[0].item()}/{totals[1].item()}",
                flush=True,
            )
    for name in (
        "FUSED_A2A_V4_OUTPUT",
        "FUSED_A2A_HADAMARD",
        *(f"FUSED_A2A_CODEC_{r}" for r in "QKV"),
    ):
        os.environ.pop(name, None)


def _check_v4_v_output(rank, world_size, device):
    from aiter.ops.mha_v4 import quantize_v_mxfp4

    heads, head_dim = 8, 128
    local_heads = heads // world_size
    os.environ["FUSED_A2A_V4_OUTPUT"] = "0"
    os.environ["FUSED_A2A_V4_OUTPUT_V"] = "1"
    for role in "QKV":
        os.environ[f"FUSED_A2A_CODEC_{role}"] = "mxfp4"
    # Rank boundaries inside tiles, tile crossings, and persistent workgroup reuse.
    for seq_len, fused, block_num in (
        (32, False, 128),
        (96, True, 128),
        (160, False, 2),
    ):
        seq_full = seq_len * world_size
        tiles = (seq_full + 127) // 128
        inputs = [
            _sequence_major_input(rank + 23 * role, heads, seq_len, head_dim, device)
            for role in range(3)
        ]
        tokens = torch.arange(seq_len, device=device).view(1, -1, 1, 1)
        channels = torch.arange(head_dim, device=device).view(1, 1, 1, -1)
        head_ids = torch.arange(heads, device=device).view(1, 1, -1, 1)
        values = (
            (tokens * 13 + channels * 7 + head_ids * 11 + rank * 17) % 61 - 30
        ).float()
        values *= torch.exp2(((tokens // 32 + channels + rank) % 7 - 3).float())
        values[..., :8] = 0
        inputs[2] = values.to(torch.bfloat16).contiguous()
        gathered = [torch.empty_like(inputs[2]) for _ in range(world_size)]
        dist.all_gather(gathered, inputs[2])
        full = torch.cat(gathered, dim=1)[
            :, :, rank * local_heads : (rank + 1) * local_heads
        ].contiguous()
        expected, expected_scales = quantize_v_mxfp4(full)
        logical_amax = (
            full.float().reshape(1, seq_full // 32, 32, local_heads, 128).abs().amax(2)
        )
        h = torch.arange(local_heads, device=device).view(-1, 1, 1)
        g = torch.arange(seq_full // 32, device=device).view(1, -1, 1)
        d = torch.arange(128, device=device).view(1, 1, -1)
        scale_offsets = (
            (h * tiles + g // 4) * 512
            + (g % 4) * 128
            + (d % 32 // 2) * 8
            + d // 32
            + (d % 2) * 4
        )
        reference_amax = logical_amax[0].transpose(0, 1)
        s = torch.arange(seq_full, device=device).view(1, -1, 1)
        pair = torch.arange(64, device=device).view(1, 1, -1)
        j = s % 32
        column = (s % 64 // 32) * 32 + 4 * (j // 8) + 16 * ((j // 4) % 2) + j % 4
        payload_offsets = (
            (h * tiles + s // 128) * 8192
            + (2 * (pair // 16) + s % 128 // 64) * 1024
            + column * 16
            + pair % 16
        )
        valid_bytes = torch.zeros_like(expected, dtype=torch.bool)
        valid_bytes[payload_offsets.flatten()] = True
        norm = torch.ones(heads * head_dim, device=device, dtype=torch.bfloat16)
        cos = torch.ones((1, seq_len, 1, head_dim), device=device)
        sin = torch.zeros_like(cos)
        op = FusedA2AIntraNodeOp(
            rank=rank,
            world_size=world_size,
            shape=inputs[0].shape,
            split=True,
            quant=True,
            return_mode="fp8",
            fuse_norm_rope=fused,
            block_num=block_num,
        )
        counts = [0, 0]
        for epoch in range(3):
            payloads, scales = op(*inputs, norm, norm, cos, sin)
            torch.cuda.synchronize()
            label = f"V4 V S={seq_len} fused={fused} rank={rank} epoch={epoch}"
            assert payloads[2].shape == expected.shape, label
            assert scales[2].numel() == expected_scales.numel(), label
            actual_scale = scales[2][scale_offsets]
            reference_scale = expected_scales.flatten()[scale_offsets]
            ties, total = _assert_fused_fp6_scales(
                actual_scale, reference_scale, reference_amax, label, max_value=6.0
            )
            counts[0] += ties
            counts[1] += total
            same = (actual_scale == reference_scale).repeat_interleave(32, dim=1)
            same_pair = same[..., 0::2] & same[..., 1::2]
            _assert_equal(
                payloads[2][payload_offsets][same_pair],
                expected[payload_offsets][same_pair],
                label,
            )
            _assert_equal(
                payloads[2][~valid_bytes],
                torch.zeros_like(payloads[2][~valid_bytes]),
                f"{label} padding/slack",
            )
            dist.barrier()
        totals = torch.tensor(counts, device=device, dtype=torch.int64)
        dist.all_reduce(totals)
        if rank == 0:
            print(
                f"PASS V4 MXFP4 V S={seq_len} fused={fused} blocks={block_num} epochs=3 scale-ties={totals[0].item()}/{totals[1].item()}",
                flush=True,
            )
    for name in (
        "FUSED_A2A_V4_OUTPUT",
        "FUSED_A2A_V4_OUTPUT_V",
        *(f"FUSED_A2A_CODEC_{r}" for r in "QKV"),
    ):
        os.environ.pop(name, None)


def _check_v4_attention(rank, world_size, device, seq_len):
    from aiter.ops.mha_v4 import (
        AttentionFormat,
        AttentionScaleMode,
        mha_v4,
        mha_v4_packed,
        mxfp4_k_view,
        mxfp4_v_view,
    )
    from aiter.test_mha_common import attention_ref

    heads, head_dim = 8, 128
    local_heads = heads // world_size
    seq_full = seq_len * world_size
    tiles = (seq_full + 127) // 128
    softmax_scale = head_dim**-0.5
    settings = {"FUSED_A2A_HADAMARD": "0"}
    for role in "QKV":
        settings[f"FUSED_A2A_CODEC_{role}"] = "mxfp4"
        settings[f"FUSED_A2A_V4_OUTPUT_{role}"] = "1"
    previous = {name: os.environ.get(name) for name in settings}
    os.environ.update(settings)
    try:
        inputs = [
            _sequence_major_input(rank + 23 * role, heads, seq_len, head_dim, device)
            for role in range(3)
        ]
        for role, value in enumerate(inputs):
            value[:, 0].zero_()
            value[:, 1] = (
                (torch.arange(heads * head_dim, device=device) % 31 - 15).reshape(
                    heads, head_dim
                )
                * (rank + role + 1)
                / 32
            )
        references = []
        for source in inputs:
            gathered = [torch.empty_like(source) for _ in range(world_size)]
            dist.all_gather(gathered, source)
            references.append(
                torch.cat(gathered, dim=1)[
                    :, :, rank * local_heads : (rank + 1) * local_heads
                ].contiguous()
            )
        norm = torch.ones(heads * head_dim, device=device, dtype=torch.bfloat16)
        cos = torch.ones((1, seq_len, 1, head_dim), device=device)
        sin = torch.zeros_like(cos)
        op = FusedA2AIntraNodeOp(
            rank=rank,
            world_size=world_size,
            shape=inputs[0].shape,
            fuse_norm_rope=False,
            split=True,
            quant=True,
            return_mode="fp8",
            softmax_scale=softmax_scale,
            block_num=2 if seq_len == 160 else 128,
        )
        payloads, scales = op(*inputs, norm, norm, cos, sin)
        q_descale, k_descale = (
            scale.view(1, seq_full, local_heads, 4) for scale in scales[:2]
        )
        v_descale = scales[2].view(1, local_heads, tiles * 512)
        # These views retain the transport buffers; no consumer-side repacking.
        q = payloads[0].view(1, seq_full, local_heads, 64)
        k = mxfp4_k_view(payloads[1], k_descale)
        v = mxfp4_v_view(payloads[2], v_descale, seq_full)
        fmt = AttentionFormat.MXFP4
        mode = AttentionScaleMode.E8M0_PER_1X32
        output = torch.empty_like(references[0])
        actual = mha_v4_packed(
            q,
            k,
            v,
            q_descale,
            k_descale,
            v_descale,
            fmt,
            fmt,
            fmt,
            mode,
            mode,
            mode,
            softmax_scale=softmax_scale,
            out=output,
        )
        expected = mha_v4(*references, fmt, fmt, fmt, softmax_scale=softmax_scale)
        oracle = attention_ref(*references, causal=False, upcast=True)[0]
        reordered = attention_ref(
            *references, causal=False, upcast=False, reorder_ops=True
        )[0]
        # Forward-only equivalent of attention_ref_with_tol(is_fp8=True).
        baseline = (reordered.float() - oracle.float()).abs().max().item()
        atol, rtol = max(4 * baseline, 0.5), 0.1
        primary_error = (actual.float() - expected.float()).abs().max().item()
        oracle_error = (actual.float() - oracle.float()).abs()
        label = f"V4 attention S={seq_len} global-S={seq_full} rank={rank}"
        print(
            f"{label}: primary-max-abs={primary_error:.9g} (atol=0 rtol=0) "
            f"oracle-max-abs={oracle_error.max().item():.9g} "
            f"oracle-max-normalized={(oracle_error / (atol + rtol * oracle.float().abs())).max().item():.9g} "
            f"(atol={atol:.9g} rtol={rtol})",
            flush=True,
        )
        assert actual is output, label
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=0, rtol=0, msg=label
        )
        torch.testing.assert_close(
            actual.float(), oracle.float(), atol=atol, rtol=rtol, msg=label
        )
        dist.barrier()
        if rank == 0:
            print(f"PASS V4 MXFP4 attention S={seq_len}", flush=True)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_rank(rank, world_size, port, v4_only=False, attention_seq=None):
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
        if attention_seq is not None:
            _check_v4_attention(rank, world_size, device, attention_seq)
            return
        _check_v4_output(rank, world_size, device)
        _check_v4_output(rank, world_size, device, q_codec="mxfp6", k_codec="mxfp6")
        _check_v4_v_output(rank, world_size, device)
        if v4_only:
            return

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

            if case_name == "small":
                _check_hadamard_transport(rank, world_size, device, a2a_references)

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

            quantizers = {
                "int8": _mx_int8_reference,
                "e4m3": _mx_fp8_reference,
                "mxfp4": _mx_fp4_reference,
                "mxfp6": _mx_fp6_reference,
            }
            for row_name, codecs in _CODEC_ROWS:
                for role, role_codec in zip("QKV", codecs, strict=True):
                    os.environ[f"FUSED_A2A_CODEC_{role}"] = role_codec
                quantized = [
                    quantizers[role_codec](input_tensor)
                    for input_tensor, role_codec in zip(inputs, codecs, strict=True)
                ]
                payload_references = [
                    a2a_references(
                        [payload],
                        heads_local,
                        seq_len,
                        head_dim * {"mxfp4": 4, "mxfp6": 6}.get(role_codec, 8) // 8,
                    )[0]
                    for (payload, _), role_codec in zip(quantized, codecs, strict=True)
                ]
                scale_references = a2a_references(
                    [scale for _, scale in quantized],
                    heads_local,
                    seq_len,
                    head_dim // 32,
                )
                modes = [(True, "fp8"), (True, "bf16")]
                if case_name == "small" and row_name == "mixed":
                    modes.append((False, "bf16"))
                for split, return_mode in modes:
                    op = FusedA2AIntraNodeOp(
                        rank=rank,
                        world_size=world_size,
                        shape=q.shape,
                        fuse_norm_rope=False,
                        split=split,
                        quant=True,
                        return_mode=return_mode,
                    )
                    for epoch in range(3):
                        result = op(*inputs)
                        torch.cuda.synchronize()
                        payloads = (
                            result[0]
                            if return_mode == "fp8"
                            else op.outputs_sets[epoch % 2]
                        )
                        scales = (
                            result[1]
                            if return_mode == "fp8"
                            else op.scales_sets[epoch % 2]
                        )
                        for i, role_codec in enumerate(codecs):
                            label = f"{case_name} {row_name} {return_mode} {'qkv'[i]} rank={rank} epoch={epoch}"
                            expected_payload = payload_references[i]
                            assert (
                                payloads[i].numel() == expected_payload.numel()
                            ), label
                            assert payloads[i].dtype == torch.uint8, label
                            _assert_equal(
                                payloads[i].view_as(expected_payload),
                                expected_payload,
                                label,
                            )
                            _assert_equal(
                                scales[i].view_as(scale_references[i]),
                                scale_references[i],
                                label,
                            )
                            reference = _dequantize(
                                expected_payload, scale_references[i], codec=role_codec
                            )
                            actual = (
                                _dequantize(
                                    payloads[i],
                                    scales[i].view_as(scale_references[i]),
                                    codec=role_codec,
                                )
                                if return_mode == "fp8"
                                else result[i].view(expected_shape)
                            )
                            if return_mode == "bf16":
                                assert actual.dtype == torch.bfloat16, label
                                reference = reference.to(torch.bfloat16)
                            _assert_equal(actual, reference, label)
                            _assert_quantized_values(
                                actual, reference, transport_references[i], label
                            )
                        dist.barrier()
                    if rank == 0:
                        print(
                            f"PASS {case_name} row={row_name} codecs={codecs} split={split} "
                            f"norm_rope=False return={return_mode}: payload/scales=byte-identical "
                            "dequant=bit-exact epochs=3",
                            flush=True,
                        )
            for row_name, codecs in _CODEC_ROWS[-2:]:
                for role, role_codec in zip("QKV", codecs, strict=True):
                    os.environ[f"FUSED_A2A_CODEC_{role}"] = role_codec
                fused_inputs = (
                    _norm_rope(q, norm_q, cos, sin, torch.float32),
                    _norm_rope(k, norm_k, cos, sin, torch.float32),
                    v,
                )
                quantized = [
                    quantizers[role_codec](value)
                    for value, role_codec in zip(fused_inputs, codecs, strict=True)
                ]
                quant_references = a2a_references(
                    [
                        _dequantize(payload, scale, codec=role_codec)
                        for (payload, scale), role_codec in zip(
                            quantized, codecs, strict=True
                        )
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
                amax_references = a2a_references(
                    [
                        value.float().reshape(*value.shape[:-1], -1, 32).abs().amax(-1)
                        for value in fused_inputs[:2]
                    ],
                    heads_local,
                    seq_len,
                    head_dim // 32,
                )
                for split, return_mode in ((False, "fp8"), (True, "bf16")):
                    tie_counts = [0, 0]
                    op = FusedA2AIntraNodeOp(
                        rank=rank,
                        world_size=world_size,
                        shape=q.shape,
                        fuse_norm_rope=True,
                        split=split,
                        quant=True,
                        return_mode=return_mode,
                    )
                    for epoch in range(3):
                        result = op(*inputs, norm_q, norm_k, cos, sin)
                        torch.cuda.synchronize()
                        payloads = (
                            result[0]
                            if return_mode == "fp8"
                            else op.outputs_sets[epoch % 2]
                        )
                        scales = (
                            result[1]
                            if return_mode == "fp8"
                            else op.scales_sets[epoch % 2]
                        )
                        for i, role_codec in enumerate(codecs):
                            label = f"{case_name} fused {row_name} {return_mode} {'qkv'[i]} rank={rank} epoch={epoch}"
                            if i < 2:
                                count, total = _assert_fused_fp6_scales(
                                    scales[i].view_as(scale_references[i]),
                                    scale_references[i],
                                    amax_references[i],
                                    label,
                                )
                                tie_counts[0] += count
                                tie_counts[1] += total
                            else:
                                _assert_equal(
                                    scales[i].view_as(scale_references[i]),
                                    scale_references[i],
                                    label,
                                )
                            decoded = _dequantize(
                                payloads[i],
                                scales[i].view_as(scale_references[i]),
                                codec=role_codec,
                            )
                            actual = (
                                decoded
                                if return_mode == "fp8"
                                else result[i].view(expected_shape)
                            )
                            reference = quant_references[i]
                            if return_mode == "bf16":
                                _assert_quantized_values(
                                    decoded, reference, references[i], label
                                )
                                _assert_equal(actual, decoded.to(torch.bfloat16), label)
                                reference = reference.to(torch.bfloat16)
                            _assert_quantized_values(
                                actual, reference, references[i], label
                            )
                        dist.barrier()
                    counts = torch.tensor(tie_counts, dtype=torch.int64, device=device)
                    dist.all_reduce(counts)
                    if rank == 0:
                        ties, blocks_checked = counts.tolist()
                        print(
                            f"PASS {case_name} row={row_name} split={split} norm_rope=True return={return_mode}: "
                            f"format-correct epochs=3 scale-ties={ties}/{blocks_checked} "
                            f"fraction={ties / blocks_checked:.9g}",
                            flush=True,
                        )
            for role in "QKV":
                os.environ.pop(f"FUSED_A2A_CODEC_{role}", None)

            if world_size >= 2:
                out_input = references[0].contiguous()
                out_packed = out_input.permute(2, 0, 1, 3).contiguous()
                out_reference = funcol.wait_tensor(
                    funcol.all_to_all_single(
                        out_packed.view(-1), None, None, dist.group.WORLD
                    )
                ).view(world_size, seq_len, heads_local, head_dim)
                # The wrapper returns [B,H_total,S_local,D]; each received peer
                # chunk is sequence-major and must be transposed before joining heads.
                out_reference = out_reference.permute(0, 2, 1, 3).reshape(
                    1, heads, seq_len, head_dim
                )
                _assert_equal(
                    out_reference.transpose(1, 2),
                    transformed[0],
                    f"{case_name} inverse reference rank {rank}",
                )
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

    _check_hadamard_precision()
    _check_fp6_converters()
    mp.spawn(_run_rank, args=(_WORLD_SIZE, _free_port()), nprocs=_WORLD_SIZE, join=True)
    passed = 16 + sum(
        12 + 3 * (name == "small") + int(_WORLD_SIZE >= 2) + 2 * len(_CODEC_ROWS)
        for name, *_ in _CASES
    )
    skipped = len(_CASES) * int(_WORLD_SIZE < 2)
    print(f"{passed} passed, {skipped} skipped on {arch}")
    return 0


@pytest.mark.parametrize("seq_len", (32, 96, 160))
def test_fused_a2a_v4_attention(capfd, seq_len):
    if not torch.cuda.is_available():
        pytest.skip("fused_a2a requires ROCm GPUs")
    if get_gfx_runtime() != "gfx950":
        pytest.skip("MHA V4 MXFP4 requires gfx950")
    if _WORLD_SIZE != 4:
        pytest.skip("MHA V4 attention integration requires world_size=4")
    if torch.cuda.device_count() < _WORLD_SIZE:
        pytest.skip(f"fused_a2a requires {_WORLD_SIZE} visible GPUs")
    # Equal local lengths divisible by 32 at ws=4 cannot produce a partial V tile.
    with capfd.disabled():
        mp.spawn(
            _run_rank,
            args=(_WORLD_SIZE, _free_port(), False, seq_len),
            nprocs=_WORLD_SIZE,
            join=True,
        )


def test_fused_a2a_v4_output(capfd):
    if not torch.cuda.is_available():
        pytest.skip("fused_a2a requires ROCm GPUs")
    if get_gfx_runtime() != "gfx950":
        pytest.skip("MHA V4 MXFP4 requires gfx950")
    if torch.cuda.device_count() < _WORLD_SIZE:
        pytest.skip(f"fused_a2a requires {_WORLD_SIZE} visible GPUs")
    with capfd.disabled():
        mp.spawn(
            _run_rank,
            args=(_WORLD_SIZE, _free_port(), True),
            nprocs=_WORLD_SIZE,
            join=True,
        )


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
