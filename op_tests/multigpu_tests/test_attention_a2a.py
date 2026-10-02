# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Multi-rank correctness and timing for attention intranode A2A transport.

One row per (world_size, recipe, sequence, heads). Every row checks the transport
against a torch/packer reference and then times the exchange with run_perftest.
"""

from __future__ import annotations

import argparse
import functools
import gc
import itertools
import math
import os
import socket
from multiprocessing import freeze_support, get_context
from types import SimpleNamespace
from typing import NamedTuple

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.mha_v4 import AttentionPack
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility.fp4_utils import _f32_to_floatx_unpacked, e8m0_to_f32

SUPPORTED_GFX = ["gfx942", "gfx950"]


class _PackedCase(NamedTuple):
    name: str
    qk_codec: str
    v_codec: str
    v_pack: AttentionPack = AttentionPack.DEFAULT


_NATIVE_CASES = (
    ("int8", "int8"),
    ("e4m3-mxfp4", ("e4m3", "mxfp4")),
    ("mxfp4-e4m3", ("mxfp4", "e4m3")),
    ("mxfp6-mxfp4", ("mxfp6", "mxfp4")),
    ("mxfp6-mxfp6", ("mxfp6", "mxfp6")),
)
_PACKED_CASES = (
    _PackedCase("int8-e4m3", "int8", "e4m3"),
    _PackedCase("e4m3-e4m3", "e4m3", "e4m3"),
    _PackedCase("mxfp4-mxfp4", "mxfp4", "mxfp4"),
    _PackedCase("mxfp6-mxfp4", "mxfp6", "mxfp4"),
    _PackedCase("mxfp6-mxfp4-fp6p", "mxfp6", "mxfp4", AttentionPack.V_FOR_FP6_P),
    _PackedCase("e4m3-mxfp6_p", "e4m3", "mxfp6_p", AttentionPack.V_FOR_FP6_P),
)
RECIPES = (
    "bf16",
    *(f"native-{name}" for name, _ in _NATIVE_CASES),
    *(f"packed-{case.name}" for case in _PACKED_CASES),
    "fp6p-mxfp4",
    "fp6p-mxfp6_p",
    "hadamard",
    "reuse",
    "e4m3_pc",
    "v4-consumer-mxfp6_p",
    "v4-consumer-mxfp4",
)
_SEQUENCE_DEPENDENT = (
    "packed-mxfp6-mxfp4-fp6p",
    "packed-e4m3-mxfp6_p",
    "fp6p-mxfp4",
    "fp6p-mxfp6_p",
    "e4m3_pc",
    "v4-consumer-mxfp6_p",
    "v4-consumer-mxfp4",
)
# gfx942 has no packed FP6 Q/K oracle, so it runs the BF16, native and first
# two packed recipes only.
_GFX942_RECIPES = (
    "bf16",
    *(f"native-{name}" for name, _ in _NATIVE_CASES),
    "packed-int8-e4m3",
    "packed-e4m3-e4m3",
    "hadamard",
    "reuse",
)

_CTX = SimpleNamespace()
_STATS = SimpleNamespace(checks=0, err=0.0)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _as_float(tensor):
    if tensor.element_size() == 1:
        tensor = tensor.view(torch.uint8)
    return tensor.float().reshape(-1)


def _exact(actual, expected, label):
    """Zero-tolerance comparison: any differing element fails the run."""
    assert (
        actual.shape == expected.shape
    ), f"{label}: {actual.shape} vs {expected.shape}"
    a, b = _as_float(actual), _as_float(expected)
    if checkAllclose(a, b, rtol=0, atol=0, printLog=False) != 0:
        checkAllclose(a, b, rtol=0, atol=0, msg=f"{label}: ")
        raise AssertionError(f"{label}: mismatch")
    _STATS.checks += 1


def _supported(recipe, world_size, sequence, heads, gfx, first_sequence):
    if heads % world_size:
        return False
    if gfx == "gfx942" and recipe not in _GFX942_RECIPES:
        return False
    # Only the FP6-P layouts depend on the sequence modulus; every other recipe
    # is sequence-independent and runs at the first sequence only.
    if recipe not in _SEQUENCE_DEPENDENT and sequence != first_sequence:
        return False
    if recipe.startswith(("packed-", "fp6p-", "v4-consumer")) and sequence % 32:
        return False
    if recipe.startswith("v4-consumer"):
        # The dense MHA V4 consumer requires world_size * sequence % 128 == 0.
        return world_size * sequence % 128 == 0
    return True


def _input(rank, heads, sequence, device, salt=0, outliers=True):
    generator = torch.Generator(device="cpu").manual_seed(1103 + rank + salt)
    value = torch.randn((1, sequence, heads, 128), generator=generator)
    value = value.to(torch.bfloat16).to(device)
    if outliers:
        value[:, 0].zero_()
        value[:, -1, :, 13] = (rank + salt + 1) * 3.25
    return value.contiguous()


def _submit_roles(op, inputs):
    for role, value in enumerate(inputs):
        result = op.submit_role(role, value)
    return result


def _from_ranks(tensors, rank, world_size):
    heads_local = tensors[0].shape[2] // world_size
    return (
        torch.cat(tensors, dim=1)[:, :, rank * heads_local : (rank + 1) * heads_local]
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def _all_to_all(tensor, rank, world_size):
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return _from_ranks(gathered, rank, world_size)


def _full_bshd(value, rank, world_size):
    return _all_to_all(value, rank, world_size).permute(0, 2, 1, 3).contiguous()


def _native_reference(codec, value, fp8_dtype, fp8_max):
    if codec == "int8":
        return _mx_int8_reference(value)
    if codec == "e4m3":
        return _mx_fp8_reference(value, fp8_dtype, fp8_max)
    if codec == "mxfp4":
        return _mx_fp4_reference(value)
    if codec == "mxfp6":
        return _mx_fp6_reference(value)
    raise AssertionError(f"unknown codec {codec}")


def _e8m0_reciprocal(exponent):
    return ((254 - exponent) << 23).view(torch.float32)


def _mx_fp8_reference(values, fp8_dtype, fp8_max):
    blocks = values.float().reshape(*values.shape[:-1], -1, 32)
    amax = blocks.abs().amax(dim=-1)
    inv_max = torch.tensor(1.0 / fp8_max, dtype=torch.float32, device=values.device)
    bits = (amax * inv_max).view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).int()
    reciprocal = _e8m0_reciprocal(exponent.clamp(0, 255))
    payload = (blocks * reciprocal.unsqueeze(-1)).to(fp8_dtype)
    return payload.view(torch.uint8).reshape_as(values), exponent.to(torch.uint8)


def _mx_int8_reference(values):
    blocks = values.float().reshape(*values.shape[:-1], -1, 32)
    need = (blocks.abs().amax(dim=-1) / 127.0).clamp_min(1.0e-30)
    exponent = (torch.ceil(torch.log2(need)) + 127.0).clamp(0, 254)
    scale = torch.exp2(exponent - 127.0)
    payload = torch.round(blocks / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return payload.view(torch.uint8).reshape_as(values), exponent.to(torch.uint8)


def _mx_fp4_reference(values):
    blocks = values.float().reshape(*values.shape[:-1], -1, 32)
    amax = blocks.abs().amax(dim=-1)
    inv_max = torch.tensor(0x3E2AAAAB, dtype=torch.int32, device=values.device).view(
        torch.float32
    )
    bits = (amax * inv_max).view(torch.int32)
    exponent = (((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).int()).clamp(0, 255)
    scaled = blocks * _e8m0_reciprocal(exponent).unsqueeze(-1)
    nibble = _f32_to_floatx_unpacked(scaled, 2, 1).reshape_as(values)
    return nibble[..., 0::2] | (nibble[..., 1::2] << 4), exponent.to(torch.uint8)


def _fp6_levels(device):
    codes = torch.arange(32, device=device)
    exponent, mantissa = codes >> 3, codes & 7
    return torch.where(
        exponent == 0,
        mantissa.float() * 2.0**-3,
        (1 + mantissa.float() / 8) * torch.exp2(exponent.float() - 1),
    )


def _mx_fp6_reference(values):
    blocks = values.float().reshape(*values.shape[:-1], -1, 32)
    bits = (blocks.abs().amax(dim=-1) * (1.0 / 7.5)).view(torch.int32)
    exponent = (((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).int()).clamp(0, 255)
    scaled = blocks * _e8m0_reciprocal(exponent).unsqueeze(-1)
    codes = _f32_to_floatx_unpacked(scaled, 2, 3).reshape(-1, 4).int()
    words = codes[:, 0] | (codes[:, 1] << 6) | (codes[:, 2] << 12) | (codes[:, 3] << 18)
    payload = torch.stack((words & 255, (words >> 8) & 255, words >> 16), dim=-1)
    return payload.to(torch.uint8).reshape(
        *values.shape[:-1], values.shape[-1] * 3 // 4
    ), exponent.to(torch.uint8)


def _dequantize(payload, scales, codec, fp8_dtype):
    if codec == "mxfp6":
        triplets = payload.reshape(-1, 3).int()
        words = triplets[:, 0] | (triplets[:, 1] << 8) | (triplets[:, 2] << 16)
        codes = torch.stack([(words >> (6 * i)) & 63 for i in range(4)], dim=-1)
        levels = _fp6_levels(payload.device)
        values = torch.where(
            codes & 32 != 0, -levels[(codes & 31).long()], levels[(codes & 31).long()]
        )
    elif codec == "mxfp4":
        nibbles = torch.stack((payload & 15, payload >> 4), dim=-1).long()
        levels = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=payload.device
        )
        values = levels[nibbles & 7] * torch.where(nibbles & 8 != 0, -1.0, 1.0)
    else:
        values = payload.view(torch.int8 if codec == "int8" else fp8_dtype).float()
    values = values.reshape(*scales.shape, 32)
    return (values * e8m0_to_f32(scales).unsqueeze(-1)).flatten(-2)


def _codecs(pair):
    return (pair, pair, pair) if isinstance(pair, str) else (pair[0], pair[0], pair[1])


def _check_bf16(rank, world_size, heads, sequence, device, op_cls):
    inputs = tuple(
        _input(rank + 19 * role, heads, sequence, device) for role in range(3)
    )
    expected = tuple(_all_to_all(value, rank, world_size) for value in inputs)
    op = op_cls(rank=rank, world_size=world_size, shape=inputs[0].shape)
    actual = _submit_roles(op, inputs)
    torch.cuda.synchronize()
    for role, (got, want) in enumerate(zip(actual, expected, strict=True)):
        _exact(got.view_as(want), want, f"bf16 {'QKV'[role]}")
    return op, lambda: _submit_roles(op, inputs)


def _check_native(rank, world_size, heads, sequence, device, op_cls, name):
    pair = dict(_NATIVE_CASES)[name]
    inputs = tuple(
        _input(rank + 31 * role, heads, sequence, device) for role in range(3)
    )
    codecs = _codecs(pair)
    op = op_cls(
        rank=rank,
        world_size=world_size,
        shape=inputs[0].shape,
        quant=pair,
        hadamard=False,
    )
    actual = _submit_roles(op, inputs)
    torch.cuda.synchronize()
    for role, (codec, value, got) in enumerate(
        zip(codecs, inputs, actual, strict=True)
    ):
        payload, scales = _native_reference(codec, value, _CTX.fp8_dtype, _CTX.fp8_max)
        payload = _all_to_all(payload, rank, world_size)
        scales = _all_to_all(scales, rank, world_size)
        label = f"{name} {'QKV'[role]}"
        _exact(op.outputs_sets[0][role].view_as(payload), payload, f"{label} payload")
        _exact(op.scales_sets[0][role].view_as(scales), scales, f"{label} scales")
        _exact(
            got.view(-1),
            _dequantize(payload, scales, codec, _CTX.fp8_dtype)
            .to(torch.bfloat16)
            .reshape(-1),
            f"{label} dequantized bf16",
        )
    return op, lambda: _submit_roles(op, inputs)


def _assert_scale_backing(scale, required, label):
    backed = scale.untyped_storage().nbytes() - scale.storage_offset()
    assert backed >= required, f"{label}: backed={backed}, required={required}"
    backing = torch.empty(0, dtype=scale.dtype, device=scale.device).set_(
        scale.untyped_storage(), scale.storage_offset(), (backed,), (1,)
    )
    _exact(
        backing[scale.numel() :],
        torch.zeros_like(backing[scale.numel() :]),
        f"{label} scale slack",
    )


def _valid_mxfp4_k_mask(raw, scales):
    batch, sequence, heads, _ = scales.shape
    tiles = (sequence + 127) // 128
    # K bytes are chunk-interleaved in each physical 128-token tile.
    b = torch.arange(batch, device=raw.device).view(-1, 1, 1, 1, 1)
    h = torch.arange(heads, device=raw.device).view(1, -1, 1, 1, 1)
    s = torch.arange(sequence, device=raw.device).view(1, 1, -1, 1, 1)
    chunk = torch.arange(4, device=raw.device).view(1, 1, 1, -1, 1)
    j = torch.arange(16, device=raw.device).view(1, 1, 1, 1, -1)
    offsets = (
        ((b * heads + h) * tiles + s // 128) * 8192 + chunk * 2048 + s % 128 * 16 + j
    )
    mask = torch.zeros_like(raw, dtype=torch.bool)
    mask[offsets.flatten()] = True
    return mask


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
    _exact(actual[offsets], expected[offsets], f"{label} payload")
    slot = s % 32 // 16 * 256 + (s % 16 * 4 + s % 128 // 32) * 4
    first = (base + 16384 + slot + g).squeeze(-1)
    second = first + 512
    logical = scales[0]
    shifted = torch.zeros_like(logical)
    shifted[..., :3] = logical[..., 1:]
    shifted[:-1, :, 3] = logical[1:, :, 0]
    for offsets, values, region in ((first, logical, "A"), (second, shifted, "B")):
        valid[offsets.flatten()] = True
        _exact(actual[offsets], values, f"{label} tail {region}")
        _exact(actual[offsets], expected[offsets], f"{label} tail {region} packer")
    _exact(actual[~valid], torch.zeros_like(actual[~valid]), f"{label} padding")


def _packed_reference(codec, role, value, softmax_scale, packers):
    if role == 0:
        return (
            packers[codec][0](value, softmax_scale * math.log2(math.e))
            if codec.startswith("mxfp")
            else packers[codec][0](value)
        )
    if role == 1:
        return packers[codec][1](value)
    return packers[codec][2](value)


def _check_packed(rank, world_size, heads, sequence, device, op_cls, name):
    from aiter.ops.mha_v4 import (
        mxfp6_k_view,
        quantize_fp8,
        quantize_fp8_rotated,
        quantize_int8,
        quantize_mxfp4_k,
        quantize_mxfp4_q,
        quantize_mxfp6_k,
        quantize_mxfp6_q,
        quantize_v_mxfp4,
    )
    from aiter.ops.mha_v4_quant import (
        MHA_V4_KV_SCALE_LOOKAHEAD_ROWS,
        MHA_V4_KV_TILE_ROWS,
        MHA_V4_MXFP4_K_SCALE_SLACK_BYTES,
        MHA_V4_MXFP4_V_SCALE_SLACK_BYTES,
        MHA_V4_QUERY_TILE_ROWS,
        quantize_v_mxfp4_fp6_p,
        quantize_v_mxfp6_fp6_p,
    )

    case = next(c for c in _PACKED_CASES if c.name == name)
    _, qk_codec, v_codec, v_pack = case
    inputs = tuple(
        _input(rank + 47 * role, heads, sequence, device) for role in range(3)
    )
    softmax_scale = 0.125
    heads_local = heads // world_size
    seq_full = sequence * world_size
    packers = {
        "int8": (lambda value: quantize_int8(value), quantize_int8, quantize_fp8),
        "e4m3": (
            lambda value: quantize_fp8_rotated(value),
            quantize_fp8_rotated,
            quantize_fp8,
        ),
        "mxfp4": (quantize_mxfp4_q, quantize_mxfp4_k, quantize_v_mxfp4),
        "mxfp6": (quantize_mxfp6_q, quantize_mxfp6_k, None),
    }
    codecs = (qk_codec, qk_codec, v_codec)
    references = []
    for role, (codec, value) in enumerate(zip(codecs, inputs, strict=True)):
        full = _full_bshd(value, rank, world_size)
        references.append(
            (
                quantize_v_mxfp6_fp6_p(full)
                if codec == "mxfp6_p"
                else quantize_v_mxfp4_fp6_p(full)
            )
            if role == 2 and v_pack == AttentionPack.V_FOR_FP6_P
            else _packed_reference(codec, role, full, softmax_scale, packers)
        )
    op = op_cls(
        rank=rank,
        world_size=world_size,
        shape=inputs[0].shape,
        quant=(qk_codec, v_codec),
        v_pack=v_pack,
        return_packed=True,
        softmax_scale=softmax_scale,
        hadamard=True,
    )
    payloads, scales = _submit_roles(op, inputs)
    torch.cuda.synchronize()

    for role in (0, 1):
        expected_payload, expected_scales = references[role]
        label = f"{name} {'QK'[role]}"
        codec = codecs[role]
        if role == 0 and codec.startswith("mxfp"):
            padded_rows = (
                (seq_full + MHA_V4_QUERY_TILE_ROWS - 1) // MHA_V4_QUERY_TILE_ROWS
            ) * MHA_V4_QUERY_TILE_ROWS - seq_full
            _assert_scale_backing(
                scales[role],
                expected_scales.numel() + padded_rows * heads_local * 4,
                label,
            )
        if role == 1 and codec == "mxfp6":
            _, logical_scales = mxfp6_k_view(
                expected_payload, expected_scales, 1, seq_full, heads_local
            )
            _exact(
                scales[role][: logical_scales.numel()].view_as(logical_scales),
                logical_scales,
                f"{label} scales",
            )
            _check_v4_fp6_k_bytes(
                payloads[role], expected_payload, logical_scales, label
            )
            _exact(
                scales[role][logical_scales.numel() :],
                torch.zeros_like(scales[role][logical_scales.numel() :]),
                f"{label} scale slack",
            )
        elif role == 1 and codec == "mxfp4":
            padded_rows = (
                ((seq_full + MHA_V4_KV_TILE_ROWS - 1) // MHA_V4_KV_TILE_ROWS)
                * MHA_V4_KV_TILE_ROWS
                + MHA_V4_KV_SCALE_LOOKAHEAD_ROWS
                - seq_full
            )
            _assert_scale_backing(
                scales[role],
                expected_scales.numel()
                + padded_rows * heads_local * 4
                + MHA_V4_MXFP4_K_SCALE_SLACK_BYTES,
                label,
            )
            actual_scales = scales[role].view_as(expected_scales)
            _exact(actual_scales, expected_scales, f"{label} scales")
            valid = _valid_mxfp4_k_mask(payloads[role], actual_scales)
            _exact(payloads[role][valid], expected_payload[valid], f"{label} payload")
            _exact(
                payloads[role][~valid],
                torch.zeros_like(payloads[role][~valid]),
                f"{label} padding",
            )
        else:
            _exact(
                payloads[role],
                expected_payload.view(torch.uint8).flatten(),
                f"{label} payload",
            )
            _exact(
                scales[role].view_as(expected_scales),
                expected_scales,
                f"{label} scales",
            )

    expected_payload, expected_scales = references[2]
    if v_codec == "e4m3":
        _exact(
            payloads[2],
            expected_payload.view(torch.uint8).flatten(),
            f"{name} V payload",
        )
        _exact(scales[2], expected_scales.flatten(), f"{name} V scales")
    elif v_codec == "mxfp6_p":
        _exact(
            payloads[2],
            expected_payload.as_strided(payloads[2].shape, (1,)),
            f"{name} V payload and slack",
        )
        _exact(scales[2].view_as(expected_scales), expected_scales, f"{name} V scales")
    else:
        _assert_scale_backing(
            scales[2],
            expected_scales.numel() + MHA_V4_MXFP4_V_SCALE_SLACK_BYTES,
            f"{name} V",
        )
        padded = ((seq_full + 127) // 128) * 128
        payload_bytes = heads_local * padded * 64
        # V tokens are permuted across the full tile; invalid tokens are zero.
        _exact(
            payloads[2][:payload_bytes],
            expected_payload[:payload_bytes],
            f"{name} V payload",
        )
        _exact(scales[2].view_as(expected_scales), expected_scales, f"{name} V scales")
        _exact(
            payloads[2][payload_bytes:],
            torch.zeros_like(payloads[2][payload_bytes:]),
            f"{name} V slack",
        )
    return op, lambda: _submit_roles(op, inputs)


# Per-codec (relL2 max, output/reference norm-ratio range) against fp32 SDPA on
# Gaussian inputs; MXFP4 V has fewer mantissa bits, so its bounds are looser.
# The cosine floor is derived, not independent: with the norm ratio near 1,
# cos = 1 - relL2**2 / 2, so 1 - relL2_max**2 / 2 is the cosine implied by the
# relL2 limit and a fixed tighter floor would only reject in-spec codec error.
_V4_CONSUMER_LIMITS = {
    "mxfp6_p": (0.092, (0.97, 1.03)),
    "mxfp4": (0.16, (0.96, 1.03)),
}


def _check_v4_consumer(rank, world_size, heads, sequence, device, op_cls, codec):
    """Feed packed FP6-P A2A outputs through the documented views into dense mha_v4_packed."""
    import torch.nn.functional as F

    from aiter.ops.mha_v4 import (
        AttentionFormat,
        AttentionScaleMode,
        mha_v4,
        mha_v4_packed,
    )
    from aiter.ops.mha_v4_quant import mxfp4_v_view, mxfp6_k_view, mxfp6_v_tiles

    e8 = AttentionScaleMode.E8M0_PER_1X32
    softmax_scale = 0.125
    seq_full = sequence * world_size
    hl = heads // world_size

    def t(x):
        return x.transpose(1, 2).float()

    inputs = tuple(
        _input(rank + 47 * role, heads, sequence, device, outliers=False)
        for role in range(3)
    )
    full = [_full_bshd(value, rank, world_size) for value in inputs]
    ref = F.scaled_dot_product_attention(
        t(full[0]), t(full[1]), t(full[2]), scale=softmax_scale
    ).transpose(1, 2)
    op = op_cls(
        rank=rank,
        world_size=world_size,
        shape=inputs[0].shape,
        quant=("mxfp6", codec),
        return_packed=True,
        v_pack=AttentionPack.V_FOR_FP6_P,
        softmax_scale=softmax_scale,
        hadamard=True,
    )
    payloads, scales = _submit_roles(op, inputs)
    torch.cuda.synchronize()
    q = payloads[0].view(1, seq_full, hl, 96)
    q_scale = scales[0].view(1, seq_full, hl, 4)
    k, k_scale = mxfp6_k_view(payloads[1], scales[1], 1, seq_full, hl)
    if codec == "mxfp6_p":
        tiles = mxfp6_v_tiles(seq_full)
        v = payloads[2].as_strided(
            (1, seq_full, hl, 128), (hl * tiles * 12288, 96, tiles * 12288, 1)
        )
        v_scale = scales[2].view(1, hl, tiles * 512)
        v_format = AttentionFormat.MXFP6
    else:
        v_scale = scales[2].view(1, hl, -1)
        v = mxfp4_v_view(payloads[2], v_scale, seq_full)
        v_format = AttentionFormat.MXFP4
    # Transport check: the fused A2A output, consumed through the packed views,
    # must equal canonical quantize + MHA on this rank's own gathered inputs.
    # Raw formats select canonical packing, so mha_v4 takes no v_pack.
    canonical = mha_v4(
        full[0],
        full[1],
        full[2],
        AttentionFormat.MXFP6,
        AttentionFormat.MXFP6,
        v_format,
        softmax_scale=softmax_scale,
    )
    out = mha_v4_packed(
        q,
        k,
        v,
        q_scale,
        k_scale,
        v_scale,
        AttentionFormat.MXFP6,
        AttentionFormat.MXFP6,
        v_format,
        e8,
        e8,
        e8,
        v_pack=AttentionPack.V_FOR_FP6_P,
        softmax_scale=softmax_scale,
    )
    torch.cuda.synchronize()
    label = f"rank {rank} v4-consumer {codec}"
    assert torch.equal(out, canonical), (
        f"{label}: fused != canonical, max abs "
        f"{(out.float() - canonical.float()).abs().max().item()}"
    )
    a, b = out.float().flatten(), ref.flatten()
    cos = (torch.dot(a, b) / (a.norm() * b.norm())).item()
    rel = ((a - b).norm() / b.norm()).item()
    norm = (a.norm() / b.norm()).item()
    rel_max, (norm_lo, norm_hi) = _V4_CONSUMER_LIMITS[codec]
    cos_min = 1 - rel_max**2 / 2
    assert cos > cos_min, f"{label}: cosine {cos} <= {cos_min}"
    assert rel < rel_max, f"{label}: relL2 {rel} >= {rel_max}"
    assert norm_lo < norm < norm_hi, f"{label}: norm ratio {norm}"
    _STATS.checks += 4
    _STATS.err = max(_STATS.err, rel)
    return op, lambda: _submit_roles(op, inputs)


def _check_fp6_p_v(rank, world_size, heads, sequence, device, op_cls, codec):
    from aiter.ops.mha_v4_quant import (
        MHA_V4_MXFP4_V_BUFFER_SLACK_BYTES,
        MHA_V4_MXFP4_V_SCALE_SLACK_BYTES,
        MHA_V4_MXFP6_V_BUFFER_SLACK_BYTES,
        MHA_V4_MXFP6_V_TILE_BYTES,
        quantize_v_mxfp4_fp6_p,
        quantize_v_mxfp6_fp6_p,
    )

    is_fp6 = codec == "mxfp6_p"
    quantize = quantize_v_mxfp6_fp6_p if is_fp6 else quantize_v_mxfp4_fp6_p
    tile_bytes = MHA_V4_MXFP6_V_TILE_BYTES if is_fp6 else 8192
    slack_bytes = (
        MHA_V4_MXFP6_V_BUFFER_SLACK_BYTES
        if is_fp6
        else MHA_V4_MXFP4_V_BUFFER_SLACK_BYTES
    )
    scale_slack = 0 if is_fp6 else MHA_V4_MXFP4_V_SCALE_SLACK_BYTES
    payload_bytes = (
        (heads // world_size) * ((sequence * world_size + 127) // 128) * tile_bytes
    )
    op = op_cls(
        rank=rank,
        world_size=world_size,
        shape=(1, sequence, heads, 128),
        quant=("mxfp6", codec),
        return_packed=True,
        v_pack=AttentionPack.V_FOR_FP6_P,
        softmax_scale=0.125,
        hadamard=True,
    )
    # Consecutive calls alternate the flag parity and move the group maximum
    # between the two senders that share a quantization group.
    for epoch in range(3):
        inputs = tuple(
            _input(rank + 47 * role, heads, sequence, device, salt=epoch * 101)
            for role in range(3)
        )
        if epoch == 0:
            inputs[2].zero_()
        else:
            boundary = -32 if rank % 2 == 0 else 0
            amplitude = 64.0 if rank % 2 == epoch % 2 else 2.0
            inputs[2][:, boundary : None if boundary else 32] = amplitude
            inputs[2][:, :, :, 0] = torch.finfo(torch.bfloat16).tiny
            inputs[2][:, :, :, 1] = torch.finfo(torch.bfloat16).max
        expected_payload, expected_scales = quantize(
            _full_bshd(inputs[2], rank, world_size)
        )
        payloads, scales = _submit_roles(op, inputs)
        torch.cuda.synchronize()
        label = f"{codec} FP6-P epoch={epoch}"
        if is_fp6:
            expected_payload = expected_payload.as_strided(
                (payload_bytes + slack_bytes,), (1,)
            )
        assert payloads[2].numel() == payload_bytes + slack_bytes, label
        _exact(
            payloads[2][:payload_bytes],
            expected_payload[:payload_bytes],
            f"{label} payload",
        )
        _exact(scales[2].view_as(expected_scales), expected_scales, f"{label} scales")
        _exact(
            payloads[2][payload_bytes:],
            torch.zeros_like(payloads[2][payload_bytes:]),
            f"{label} payload slack",
        )
        _assert_scale_backing(scales[2], expected_scales.numel() + scale_slack, label)
    return op, lambda: _submit_roles(op, inputs)


def _check_hadamard(rank, world_size, heads, sequence, device, op_cls):
    from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
        create_hadamard_matrix,
    )

    inputs = tuple(
        _input(rank + 71 * role, heads, sequence, device) for role in range(3)
    )
    kwargs = {
        "rank": rank,
        "world_size": world_size,
        "shape": inputs[0].shape,
        "quant": ("mxfp4", "e4m3"),
    }
    off = op_cls(hadamard=False, **kwargs)
    on = op_cls(hadamard=True, **kwargs)
    _submit_roles(off, inputs)
    _submit_roles(on, inputs)
    torch.cuda.synchronize()
    for field in ("outputs_sets", "scales_sets"):
        _exact(
            getattr(off, field)[0][2], getattr(on, field)[0][2], f"hadamard V {field}"
        )
    matrix = create_hadamard_matrix(128, device=device, dtype=torch.float32)
    for role, value in enumerate(inputs[:2]):
        rotated = _all_to_all(value.float() @ matrix * 128**-0.5, rank, world_size)
        payload, scales = _mx_fp4_reference(rotated)
        expected = _dequantize(payload, scales, "mxfp4", _CTX.fp8_dtype)
        actual = _dequantize(
            on.outputs_sets[0][role].view_as(payload),
            on.scales_sets[0][role].view_as(scales),
            "mxfp4",
            _CTX.fp8_dtype,
        )
        # The kernel's butterfly and the reference matmul accumulate the FP32
        # rotation in different orders, so a rare element rounds differently.
        err = checkAllclose(
            actual.to(torch.bfloat16).float(),
            expected.to(torch.bfloat16).float(),
            rtol=0,
            atol=0,
            printLog=False,
        )
        assert err <= 1.0e-4, f"hadamard {'QK'[role]} mismatch_fraction={err:.6g}"
        _STATS.checks += 1
    assert not torch.equal(
        off.outputs_sets[0][0], on.outputs_sets[0][0]
    ), "hadamard Q output did not change"
    _STATS.checks += 1
    return (off, on), lambda: _submit_roles(on, inputs)


def _check_reuse(rank, world_size, heads, sequence, device, op_cls):
    op = op_cls(
        rank=rank,
        world_size=world_size,
        shape=(1, sequence, heads, 128),
        quant="e4m3",
        hadamard=False,
    )
    # Four calls run both flag parities twice; each must see only its own data.
    for call in range(4):
        inputs = tuple(
            _input(rank + 97 * role, heads, sequence, device, salt=call)
            for role in range(3)
        )
        payload, scales = _mx_fp8_reference(inputs[0], _CTX.fp8_dtype, _CTX.fp8_max)
        expected = _dequantize(
            _all_to_all(payload, rank, world_size),
            _all_to_all(scales, rank, world_size),
            "e4m3",
            _CTX.fp8_dtype,
        ).to(torch.bfloat16)
        result = _submit_roles(op, inputs)
        actual = result[0].view(1, heads // world_size, sequence * world_size, 128)
        torch.cuda.synchronize()
        _exact(actual, expected, f"reuse call {call}")
    return op, lambda: _submit_roles(op, inputs)


def _check_e4m3_pc(rank, world_size, heads, sequence, device, op_cls):
    from aiter.ops.mha_v4 import (
        AttentionFormat,
        AttentionScaleMode,
        mha_v4,
        mha_v4_packed,
        mxfp6_k_view,
    )
    from aiter.ops.mha_v4_quant import quantize_v_fp8

    hl = heads // world_size
    seq_full = sequence * world_size
    op = op_cls(
        rank=rank,
        world_size=world_size,
        shape=(1, sequence, heads, 128),
        quant=("mxfp6", "e4m3_pc"),
        return_packed=True,
        softmax_scale=0.125,
        hadamard=True,
    )
    for call in range(4):
        inputs = tuple(
            _input(rank + 47 * role, heads, sequence, device, salt=call)
            for role in range(3)
        )
        # Maxima move between ranks and decrease on parity reuse. Tail padding
        # is part of the input sequence; no sender may omit another rank's max.
        inputs[2].mul_(8.0 if rank == call % world_size and call < 2 else 0.25)
        if call < 3:
            inputs[2][..., 7] = 0
            inputs[2][:, -3:] = 0
        full_v = _full_bshd(inputs[2], rank, world_size)
        expected_payload, expected_scales = quantize_v_fp8(full_v)
        payloads, scales = _submit_roles(op, inputs)
        assert scales[2].shape == (1, hl, 128) and scales[2].is_contiguous()
        torch.cuda.synchronize()
        label = f"e4m3_pc call={call}"
        _exact(
            payloads[2],
            expected_payload.view(torch.uint8).flatten(),
            f"{label} payload",
        )
        _exact(scales[2], expected_scales, f"{label} scales")
    # The dense f6f8 attention needs world_size * sequence % 128 == 0.
    if seq_full % 128 == 0:
        full_qk = tuple(_full_bshd(value, rank, world_size) for value in inputs[:2])
        k, k_scale = mxfp6_k_view(payloads[1], scales[1], 1, seq_full, hl)
        actual = mha_v4_packed(
            payloads[0].view(1, seq_full, hl, 96),
            k,
            payloads[2].view(torch.float8_e4m3fn).view(1, seq_full, hl, 128),
            scales[0].view(1, seq_full, hl, 4),
            k_scale,
            scales[2],
            AttentionFormat.MXFP6,
            AttentionFormat.MXFP6,
            AttentionFormat.FP8,
            AttentionScaleMode.E8M0_PER_1X32,
            AttentionScaleMode.E8M0_PER_1X32,
            AttentionScaleMode.F32_PER_CHANNEL,
            softmax_scale=0.125,
        )
        expected = mha_v4(
            *full_qk,
            full_v,
            AttentionFormat.MXFP6,
            AttentionFormat.MXFP6,
            AttentionFormat.FP8,
            softmax_scale=0.125,
        )
        _exact(actual, expected, "e4m3_pc packed/raw f6f8 attention")
    return op, lambda: _submit_roles(op, inputs)


def _build_check(recipe, rank, world_size, heads, sequence, device, op_cls):
    args = (rank, world_size, heads, sequence, device, op_cls)
    if recipe == "bf16":
        return _check_bf16(*args)
    if recipe.startswith("native-"):
        return _check_native(*args, recipe[len("native-") :])
    if recipe.startswith("packed-"):
        return _check_packed(*args, recipe[len("packed-") :])
    if recipe.startswith("fp6p-"):
        return _check_fp6_p_v(*args, recipe[len("fp6p-") :])
    if recipe.startswith("v4-consumer-"):
        return _check_v4_consumer(*args, recipe[len("v4-consumer-") :])
    return {
        "hadamard": _check_hadamard,
        "reuse": _check_reuse,
        "e4m3_pc": _check_e4m3_pc,
    }[recipe](*args)


_DEFAULT_BLOCKS = 128  # AttentionA2AIntraNodeOp default block_num
_OVERSUBSCRIBED_BLOCKS = 4096
_OVERSUBSCRIBED_RECIPE = "bf16"


@benchmark()
def test_a2a(world_size, recipe, sequence, heads, blocks):
    """Check one recipe against its reference, then time the exchange itself."""
    _STATS.err = 0.0
    failure = None
    keep = exchange = None
    try:
        keep, exchange = _build_check(
            recipe, _CTX.rank, world_size, heads, sequence, _CTX.device, _CTX.op_cls
        )
    except AssertionError as exc:
        failure = exc
    # Ranks must agree on the verdict before any benchmark launch: the exchange
    # waits on every peer, so one rank tearing down on a failed check would leave
    # the rest spinning in the drain handshake forever.
    flags = torch.zeros(world_size, dtype=torch.int32)
    flags[_CTX.rank] = failure is not None
    dist.all_reduce(flags, group=_CTX.cpu_group)
    failed = flags.nonzero().flatten().tolist()
    if failed:
        if failure is not None:
            raise failure
        raise RuntimeError(
            f"{recipe} s={sequence} h={heads}: check failed on rank(s) {failed}"
        )
    # Bytes each rank contributes: Q, K and V in BF16. The transport does no
    # arithmetic, so there is no TFLOPS column.
    nbytes = 3 * sequence * heads * 128 * 2
    out, us = run_perftest(exchange, num_iters=20, use_cuda_event=True)
    del keep, exchange, out
    gc.collect()
    return {
        "gfx": _CTX.gfx,
        "a2a us": us,
        "a2a TB/s": nbytes / us / 1e6,
        "a2a err": _STATS.err,
    }


def _run_rank(rank, world_size, port, recipes, sequences, heads_list):
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "12G")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    import mori.shmem as ms

    from aiter.ops.flydsl.attention_a2a_intranode import AttentionA2AIntraNodeOp

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
        gfx = get_gfx()
        _CTX.rank, _CTX.device, _CTX.gfx = rank, device, gfx
        _CTX.cpu_group = cpu_group
        _CTX.fp8_dtype, _CTX.fp8_max = (
            (torch.float8_e4m3fnuz, 240.0)
            if gfx == "gfx942"
            else (torch.float8_e4m3fn, 448.0)
        )
        rows = []
        _CTX.op_cls = AttentionA2AIntraNodeOp
        for recipe, sequence, heads in itertools.product(
            recipes, sequences, heads_list
        ):
            if _supported(recipe, world_size, sequence, heads, gfx, sequences[0]):
                rows.append(
                    test_a2a(world_size, recipe, sequence, heads, _DEFAULT_BLOCKS)
                )
        # Forward-progress coverage: waits only read flags from an earlier
        # launch, so a grid far beyond co-residency (256 CUs on gfx950) must
        # still finish. One cheap case; skipped when the sweep excludes bf16.
        if _OVERSUBSCRIBED_RECIPE in recipes and _supported(
            _OVERSUBSCRIBED_RECIPE,
            world_size,
            sequences[0],
            heads_list[0],
            gfx,
            sequences[0],
        ):
            _CTX.op_cls = functools.partial(
                AttentionA2AIntraNodeOp, block_num=_OVERSUBSCRIBED_BLOCKS
            )
            rows.append(
                test_a2a(
                    world_size,
                    _OVERSUBSCRIBED_RECIPE,
                    sequences[0],
                    heads_list[0],
                    _OVERSUBSCRIBED_BLOCKS,
                )
            )
        return rows, _STATS.checks
    finally:
        try:
            ms.shmem_finalize()
        finally:
            dist.destroy_process_group()


def _run_world(world_size, recipes, sequences, heads_list):
    visible = torch.cuda.device_count()
    if visible < world_size:
        aiter.logger.warning(
            "requires %d visible GPUs, found %d; skipping", world_size, visible
        )
        return [], 0
    port = _free_port()
    with get_context("spawn").Pool(processes=world_size) as pool:
        results = pool.starmap(
            _run_rank,
            [
                (rank, world_size, port, recipes, sequences, heads_list)
                for rank in range(world_size)
            ],
        )
    return results[0][0], sum(checks for _, checks in results)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Attention intranode A2A correctness and timing",
    )
    parser.add_argument(
        "-w",
        "--world-sizes",
        type=int,
        nargs="*",
        choices=(2, 4, 8),
        default=[2],
        help="ranks per run",
    )
    parser.add_argument(
        "-r",
        "--recipes",
        type=str,
        nargs="*",
        choices=RECIPES,
        default=list(RECIPES),
        help="transport recipe (codec pair and check) to run",
    )
    parser.add_argument(
        "-s",
        "--sequences",
        type=int,
        nargs="*",
        default=[1184, 1888, 1152, 1200],
        help="per-rank sequence lengths (FP6-P %%32: 1184, 1888; %%64 control: 1152; e4m3_pc tile tail: 1200)",
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs="*",
        default=[8],
        help="total head counts (Wan2.2 deploys 40)",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("attention A2A needs gfx942 or gfx950; skipping")
        return 0
    rows, checks = [], 0
    for world_size in args.world_sizes:
        world_rows, world_checks = _run_world(
            world_size, args.recipes, args.sequences, args.heads
        )
        rows += world_rows
        checks += world_checks
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "attention A2A summary (markdown):\n%s", df.to_markdown(index=False)
    )
    print(f"completed rows={len(rows)} checks={checks} failures=0", flush=True)
    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
