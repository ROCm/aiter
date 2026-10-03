# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Multi-rank correctness for attention intranode A2A transport."""

from __future__ import annotations

import argparse
import math
import os
import socket
from multiprocessing import freeze_support, get_context
from typing import NamedTuple

import torch
import torch.distributed as dist

from aiter.ops.mha_v4 import AttentionPack


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
)
_PACKED_CASES = (
    _PackedCase("int8-e4m3", "int8", "e4m3"),
    _PackedCase("e4m3-e4m3", "e4m3", "e4m3"),
    _PackedCase("mxfp4-mxfp4", "mxfp4", "mxfp4"),
    _PackedCase("mxfp6-mxfp4", "mxfp6", "mxfp4"),
)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _assert_equal(actual, expected, label):
    if not torch.equal(actual, expected):
        mismatch = torch.nonzero(actual != expected, as_tuple=False)[0].flatten()
        raise AssertionError(
            f"{label}: mismatch at {mismatch.tolist()}: "
            f"actual={actual[tuple(mismatch)].item()} "
            f"expected={expected[tuple(mismatch)].item()}"
        )


def _input(rank, heads, sequence, device, salt=0):
    generator = torch.Generator(device="cpu").manual_seed(1103 + rank + salt)
    value = torch.randn((1, sequence, heads, 128), generator=generator)
    value = value.to(torch.bfloat16).to(device)
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


def _mx_fp8_reference(values, fp8_dtype, fp8_max):
    blocks = values.float().reshape(*values.shape[:-1], -1, 32)
    amax = blocks.abs().amax(dim=-1)
    inv_max = torch.tensor(1.0 / fp8_max, dtype=torch.float32, device=values.device)
    bits = (amax * inv_max).view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).int()
    reciprocal = ((254 - exponent.clamp(0, 255)) << 23).view(torch.float32)
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
    reciprocal = ((254 - exponent) << 23).view(torch.float32)
    magnitude = (blocks * reciprocal.unsqueeze(-1)).abs()
    nibble = torch.zeros_like(magnitude, dtype=torch.uint8)
    for index, midpoint in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)):
        nibble += (magnitude >= midpoint if index % 2 else magnitude > midpoint).to(
            torch.uint8
        )
    nibble |= torch.signbit(blocks * reciprocal.unsqueeze(-1)).to(torch.uint8) << 3
    nibble = nibble.reshape_as(values)
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
    reciprocal = ((254 - exponent) << 23).view(torch.float32)
    levels = _fp6_levels(values.device)
    scaled = blocks * reciprocal.unsqueeze(-1)
    magnitude = scaled.abs()
    codes = torch.zeros_like(scaled, dtype=torch.uint8)
    for index in range(31):
        midpoint = (levels[index] + levels[index + 1]) / 2
        codes += (magnitude >= midpoint if index % 2 else magnitude > midpoint).to(
            torch.uint8
        )
    codes |= torch.signbit(scaled).to(torch.uint8) << 5
    codes = codes.reshape(-1, 4).int()
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
    return (values * torch.exp2(scales.float() - 127).unsqueeze(-1)).flatten(-2)


def _check_bf16(rank, world_size, device, AttentionA2AIntraNodeOp):
    inputs = tuple(_input(rank + 19 * role, 8, 17, device) for role in range(3))
    expected = tuple(_all_to_all(value, rank, world_size) for value in inputs)
    op = AttentionA2AIntraNodeOp(
        rank=rank, world_size=world_size, shape=inputs[0].shape
    )
    actual = _submit_roles(op, inputs)
    torch.cuda.synchronize()
    for role, (got, want) in enumerate(zip(actual, expected, strict=True)):
        _assert_equal(got.view_as(want), want, f"bf16 {'QKV'[role]}")
    return 3


def _check_native(
    rank, world_size, device, AttentionA2AIntraNodeOp, fp8_dtype, fp8_max
):
    inputs = tuple(_input(rank + 31 * role, 8, 17, device) for role in range(3))
    checks = 0
    for name, pair in _NATIVE_CASES:
        codecs = (
            (pair, pair, pair) if isinstance(pair, str) else (pair[0], pair[0], pair[1])
        )
        expected = []
        for value, codec in zip(inputs, codecs, strict=True):
            payload, scales = _native_reference(codec, value, fp8_dtype, fp8_max)
            expected.append(
                (
                    _all_to_all(payload, rank, world_size),
                    _all_to_all(scales, rank, world_size),
                )
            )
        op = AttentionA2AIntraNodeOp(
            rank=rank,
            world_size=world_size,
            shape=inputs[0].shape,
            quant=pair,
            hadamard=False,
        )
        actual = _submit_roles(op, inputs)
        torch.cuda.synchronize()
        for role, (codec, got, (payload, scales)) in enumerate(
            zip(codecs, actual, expected, strict=True)
        ):
            actual_payload = op.outputs_sets[0][role].view_as(payload)
            actual_scales = op.scales_sets[0][role].view_as(scales)
            _assert_equal(actual_payload, payload, f"{name} {'QKV'[role]} payload")
            _assert_equal(actual_scales, scales, f"{name} {'QKV'[role]} scales")
            _assert_equal(
                got.view(-1),
                _dequantize(payload, scales, codec, fp8_dtype)
                .to(torch.bfloat16)
                .reshape(-1),
                f"{name} {'QKV'[role]} dequantized bf16",
            )
            checks += 3
    return checks


def _valid_mxfp4_k_mask(raw, scales):
    batch, sequence, heads, _ = scales.shape
    tiles = (sequence + 127) // 128
    mask = torch.zeros_like(raw, dtype=torch.bool)
    # K bytes are chunk-interleaved in each physical 128-token tile.
    for b in range(batch):
        for h in range(heads):
            for token in range(sequence):
                tile = token // 128
                in_tile = token % 128
                base = ((b * heads + h) * tiles + tile) * 8192
                for chunk in range(4):
                    start = base + chunk * 2048 + in_tile * 16
                    mask[start : start + 16] = True
    return mask


def _assert_scale_backing(scale, required, label):
    backed = scale.untyped_storage().nbytes() - scale.storage_offset()
    if backed < required:
        raise AssertionError(f"{label}: backed={backed}, required={required}")
    backing = torch.empty(0, dtype=scale.dtype, device=scale.device).set_(
        scale.untyped_storage(), scale.storage_offset(), (backed,), (1,)
    )
    _assert_equal(
        backing[scale.numel() :],
        torch.zeros_like(backing[scale.numel() :]),
        f"{label} scale slack",
    )


def _check_scale_storage_extents():
    """Validate the host-side scale backing arithmetic without constructing shmem tensors."""
    from aiter.ops.mha_v4_quant import (
        MHA_V4_KV_SCALE_LOOKAHEAD_ROWS,
        MHA_V4_KV_TILE_ROWS,
        MHA_V4_MXFP4_K_SCALE_SLACK_BYTES,
        MHA_V4_MXFP4_V_SCALE_SLACK_BYTES,
        MHA_V4_QUERY_TILE_ROWS,
    )
    from aiter.ops.triton.quant.mxfp6_fmha_pack import fp6_k_raw_buffer_sizes

    rows = []
    for world_size in (2, 4, 8):
        heads_local = 8 // world_size
        seq_full = 32 * world_size
        numel = 32 * 8 * 128
        cases = [
            (
                False,
                name,
                (
                    (pair, pair, pair)
                    if isinstance(pair, str)
                    else (pair[0], pair[0], pair[1])
                ),
            )
            for name, pair in _NATIVE_CASES
        ] + [
            (True, case.name, (case.qk_codec, case.qk_codec, case.v_codec))
            for case in _PACKED_CASES
        ]
        for return_packed, name, codecs in cases:
            per_tensor = tuple(
                (return_packed and role == 2 and codec == "e4m3")
                or (return_packed and role in (0, 1) and codec in ("int8", "e4m3"))
                for role, codec in enumerate(codecs)
            )
            fp6_k_scale_size = seq_full * heads_local * 4 + 64
            if return_packed and "mxfp6" in codecs:
                _, fp6_k_scale_size = fp6_k_raw_buffer_sizes(1, seq_full, heads_local)
            scale_sizes = tuple(
                (
                    1 + world_size * 128 * 4
                    if is_per_tensor
                    else (
                        heads_local * ((seq_full + 127) // 128) * 512
                        if return_packed and role == 2
                        else (
                            fp6_k_scale_size
                            if return_packed and role == 1 and codec == "mxfp6"
                            else numel // 32
                        )
                    )
                )
                for role, (codec, is_per_tensor) in enumerate(zip(codecs, per_tensor))
            )
            padding_sizes = tuple(
                (
                    (
                        (
                            (seq_full + MHA_V4_QUERY_TILE_ROWS - 1)
                            // MHA_V4_QUERY_TILE_ROWS
                        )
                        * MHA_V4_QUERY_TILE_ROWS
                        - seq_full
                    )
                    * heads_local
                    * 4
                    if return_packed and role == 0 and codec.startswith("mxfp")
                    else (
                        (
                            (
                                (seq_full + MHA_V4_KV_TILE_ROWS - 1)
                                // MHA_V4_KV_TILE_ROWS
                            )
                            * MHA_V4_KV_TILE_ROWS
                            + MHA_V4_KV_SCALE_LOOKAHEAD_ROWS
                            - seq_full
                        )
                        * heads_local
                        * 4
                        + MHA_V4_MXFP4_K_SCALE_SLACK_BYTES
                        if return_packed and role == 1 and codec == "mxfp4"
                        else (
                            MHA_V4_MXFP4_V_SCALE_SLACK_BYTES
                            if return_packed and role == 2 and codec == "mxfp4"
                            else 0
                        )
                    )
                )
                for role, codec in enumerate(codecs)
            )
            for role, (codec, size, padding) in enumerate(
                zip(codecs, scale_sizes, padding_sizes)
            ):
                backing = size + padding
                if backing <= 0 or backing < size:
                    raise AssertionError(
                        f"world={world_size} {name} {'QKV'[role]} {codec}: "
                        f"logical={size}, backing={backing}"
                    )
                rows.append((world_size, name, "QKV"[role], codec, size, backing))

    expected = {
        (8, "mxfp4-mxfp4", "K"): (1024, 2052),
        (8, "mxfp4-mxfp4", "V"): (1024, 2048),
        (2, "mxfp4-mxfp4", "Q"): (1024, 4096),
    }
    actual = {
        (world, name, role): (size, backing)
        for world, name, role, _, size, backing in rows
    }
    for key, required in expected.items():
        if actual[key] != required:
            raise AssertionError(f"{key}: got={actual[key]}, required={required}")
    return rows


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
    first = (base + 16384 + slot + g).squeeze(-1)
    second = first + 512
    logical = scales[0]
    shifted = torch.zeros_like(logical)
    shifted[..., :3] = logical[..., 1:]
    shifted[:-1, :, 3] = logical[1:, :, 0]
    for offsets, values, region in ((first, logical, "A"), (second, shifted, "B")):
        valid[offsets.flatten()] = True
        _assert_equal(actual[offsets], values, f"{label} tail {region}")
        _assert_equal(
            actual[offsets], expected[offsets], f"{label} tail {region} packer"
        )
    _assert_equal(actual[~valid], torch.zeros_like(actual[~valid]), f"{label} padding")


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


def _check_packed(rank, world_size, device, AttentionA2AIntraNodeOp, sequence, cases):
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
    from aiter.ops.mha_v4_quant import quantize_v_mxfp4_fp6_p

    inputs = tuple(_input(rank + 47 * role, 8, sequence, device) for role in range(3))
    softmax_scale = 0.125
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
    checks = 0
    for name, qk_codec, v_codec, v_pack in cases:
        codecs = (qk_codec, qk_codec, v_codec)
        references = []
        for role, (codec, value) in enumerate(zip(codecs, inputs, strict=True)):
            full = _all_to_all(value, rank, world_size).permute(0, 2, 1, 3).contiguous()
            references.append(
                quantize_v_mxfp4_fp6_p(full)
                if role == 2 and v_pack == AttentionPack.V_FOR_FP6_P
                else _packed_reference(codec, role, full, softmax_scale, packers)
            )
        op = AttentionA2AIntraNodeOp(
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
        from aiter.ops.mha_v4_quant import (
            MHA_V4_KV_SCALE_LOOKAHEAD_ROWS,
            MHA_V4_KV_TILE_ROWS,
            MHA_V4_MXFP4_K_SCALE_SLACK_BYTES,
            MHA_V4_MXFP4_V_SCALE_SLACK_BYTES,
            MHA_V4_QUERY_TILE_ROWS,
        )

        for role in (0, 1):
            expected_payload, expected_scales = references[role]
            label = f"{name} {'QK'[role]}"
            codec = codecs[role]
            if role == 0 and codec.startswith("mxfp"):
                required = (
                    expected_scales.numel()
                    + (
                        (
                            (sequence * world_size + MHA_V4_QUERY_TILE_ROWS - 1)
                            // MHA_V4_QUERY_TILE_ROWS
                        )
                        * MHA_V4_QUERY_TILE_ROWS
                        - sequence * world_size
                    )
                    * (8 // world_size)
                    * 4
                )
                _assert_scale_backing(scales[role], required, label)
            if role == 1 and codec == "mxfp6":
                _, logical_scales = mxfp6_k_view(
                    expected_payload,
                    expected_scales,
                    1,
                    sequence * world_size,
                    8 // world_size,
                )
                _assert_equal(
                    scales[role][: logical_scales.numel()].view_as(logical_scales),
                    logical_scales,
                    f"{label} scales",
                )
                _check_v4_fp6_k_bytes(
                    payloads[role], expected_payload, logical_scales, label
                )
                _assert_equal(
                    scales[role][logical_scales.numel() :],
                    torch.zeros_like(scales[role][logical_scales.numel() :]),
                    f"{label} scale slack",
                )
            elif role == 1 and codec == "mxfp4":
                required = expected_scales.numel() + (
                    (
                        (
                            (sequence * world_size + MHA_V4_KV_TILE_ROWS - 1)
                            // MHA_V4_KV_TILE_ROWS
                        )
                        * MHA_V4_KV_TILE_ROWS
                        + MHA_V4_KV_SCALE_LOOKAHEAD_ROWS
                        - sequence * world_size
                    )
                    * (8 // world_size)
                    * 4
                    + MHA_V4_MXFP4_K_SCALE_SLACK_BYTES
                )
                _assert_scale_backing(scales[role], required, label)
                actual_scales = scales[role].view_as(expected_scales)
                _assert_equal(actual_scales, expected_scales, f"{label} scales")
                valid = _valid_mxfp4_k_mask(payloads[role], actual_scales)
                _assert_equal(
                    payloads[role][valid], expected_payload[valid], f"{label} payload"
                )
                _assert_equal(
                    payloads[role][~valid],
                    torch.zeros_like(payloads[role][~valid]),
                    f"{label} padding",
                )
            else:
                _assert_equal(
                    payloads[role],
                    expected_payload.view(torch.uint8).flatten(),
                    f"{label} payload",
                )
                _assert_equal(
                    scales[role].view_as(expected_scales),
                    expected_scales,
                    f"{label} scales",
                )
            checks += 2

        expected_payload, expected_scales = references[2]
        if v_codec == "e4m3":
            _assert_equal(
                payloads[2],
                expected_payload.view(torch.uint8).flatten(),
                f"{name} V payload",
            )
            _assert_equal(scales[2], expected_scales.flatten(), f"{name} V scales")
        else:
            _assert_scale_backing(
                scales[2],
                expected_scales.numel() + MHA_V4_MXFP4_V_SCALE_SLACK_BYTES,
                f"{name} V",
            )
            actual_scales = scales[2].view_as(expected_scales)
            heads = 8 // world_size
            padded = ((sequence * world_size + 127) // 128) * 128
            payload_bytes = heads * padded * 64
            # V tokens are permuted across the full tile; invalid tokens are zero.
            _assert_equal(
                payloads[2][:payload_bytes],
                expected_payload[:payload_bytes],
                f"{name} V payload",
            )
            _assert_equal(actual_scales, expected_scales, f"{name} V scales")
            _assert_equal(
                payloads[2][payload_bytes:],
                torch.zeros_like(payloads[2][payload_bytes:]),
                f"{name} V slack",
            )
        checks += 2
    return checks


def _check_hadamard(rank, world_size, device, AttentionA2AIntraNodeOp, fp8_dtype):
    from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
        create_hadamard_matrix,
    )

    inputs = tuple(_input(rank + 71 * role, 8, 17, device) for role in range(3))
    off = AttentionA2AIntraNodeOp(
        rank=rank,
        world_size=world_size,
        shape=inputs[0].shape,
        quant=("mxfp4", "e4m3"),
        hadamard=False,
    )
    on = AttentionA2AIntraNodeOp(
        rank=rank,
        world_size=world_size,
        shape=inputs[0].shape,
        quant=("mxfp4", "e4m3"),
        hadamard=True,
    )
    _submit_roles(off, inputs)
    _submit_roles(on, inputs)
    torch.cuda.synchronize()
    for field in ("outputs_sets", "scales_sets"):
        _assert_equal(
            getattr(off, field)[0][2], getattr(on, field)[0][2], f"hadamard V {field}"
        )
    matrix = create_hadamard_matrix(128, device=device, dtype=torch.float32)
    rotated = [
        _all_to_all(value.float() @ matrix * 128**-0.5, rank, world_size)
        for value in inputs[:2]
    ]
    for role, expected in enumerate(rotated):
        payload, scales = _mx_fp4_reference(expected)
        expected_values = _dequantize(payload, scales, "mxfp4", fp8_dtype).to(
            torch.bfloat16
        )
        actual_values = _dequantize(
            on.outputs_sets[0][role].view_as(payload),
            on.scales_sets[0][role].view_as(scales),
            "mxfp4",
            fp8_dtype,
        ).to(torch.bfloat16)
        # FP32 rotation order can differ between butterfly and matmul accumulation.
        mismatch_fraction = (actual_values != expected_values).float().mean().item()
        if mismatch_fraction > 1.0e-4:
            raise AssertionError(
                f"hadamard {'QK'[role]} mismatch_fraction={mismatch_fraction:.6g}"
            )
    if torch.equal(off.outputs_sets[0][0], on.outputs_sets[0][0]):
        raise AssertionError("hadamard Q output did not change")
    return 5


def _check_reuse(rank, world_size, device, AttentionA2AIntraNodeOp, fp8_dtype, fp8_max):
    op = AttentionA2AIntraNodeOp(
        rank=rank,
        world_size=world_size,
        shape=(1, 17, 8, 128),
        quant="e4m3",
        hadamard=False,
    )
    consumers = []
    expected_values = []
    for call in range(3):
        inputs = tuple(
            _input(rank + 97 * role, 8, 17, device, salt=call) for role in range(3)
        )
        payload, scales = _mx_fp8_reference(inputs[0], fp8_dtype, fp8_max)
        payloads = [torch.empty_like(payload) for _ in range(world_size)]
        scale_rows = [torch.empty_like(scales) for _ in range(world_size)]
        dist.all_gather(payloads, payload)
        dist.all_gather(scale_rows, scales)
        expected_payload = _from_ranks(payloads, rank, world_size)
        expected_scales = _from_ranks(scale_rows, rank, world_size)
        result = _submit_roles(op, inputs)
        consumers.append(
            result[0].view(1, 8 // world_size, 17 * world_size, 128).clone()
        )
        expected_values.append(
            _dequantize(expected_payload, expected_scales, "e4m3", fp8_dtype).to(
                torch.bfloat16
            )
        )
    torch.cuda.synchronize()
    for call, (actual, expected) in enumerate(
        zip(consumers, expected_values, strict=True)
    ):
        _assert_equal(actual, expected, f"reuse call {call}")
    return len(consumers)


def _check_wan_2_2(
    rank, world_size, device, AttentionA2AIntraNodeOp, fp8_dtype, fp8_max
):
    inputs = tuple(_input(rank + 131 * role, 40, 9419, device) for role in range(3))
    checks = 0
    bf16 = AttentionA2AIntraNodeOp(
        rank=rank, world_size=world_size, shape=inputs[0].shape
    )
    actual = _submit_roles(bf16, inputs)
    torch.cuda.synchronize()
    for role, (got, value) in enumerate(zip(actual, inputs, strict=True)):
        expected = _all_to_all(value, rank, world_size)
        _assert_equal(
            got.view_as(expected),
            expected,
            f"wan_2_2 bf16 {'QKV'[role]}",
        )
        checks += 1
    quant = AttentionA2AIntraNodeOp(
        rank=rank,
        world_size=world_size,
        shape=inputs[0].shape,
        quant=("mxfp4", "e4m3"),
        hadamard=False,
    )
    actual = _submit_roles(quant, inputs)
    torch.cuda.synchronize()
    for role, (codec, value, got) in enumerate(
        zip(("mxfp4", "mxfp4", "e4m3"), inputs, actual, strict=True)
    ):
        payload, scales = _native_reference(codec, value, fp8_dtype, fp8_max)
        payload, scales = _all_to_all(payload, rank, world_size), _all_to_all(
            scales, rank, world_size
        )
        _assert_equal(
            got.view(-1),
            _dequantize(payload, scales, codec, fp8_dtype)
            .to(torch.bfloat16)
            .reshape(-1),
            f"wan_2_2 quant {'QKV'[role]}",
        )
        checks += 1
    return checks


def _run_rank(
    rank, world_size, port, run_full, run_tail, fp8_dtype, fp8_max, packed_cases
):
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
        checks = 0
        if run_full:
            checks += _check_bf16(rank, world_size, device, AttentionA2AIntraNodeOp)
            checks += _check_native(
                rank, world_size, device, AttentionA2AIntraNodeOp, fp8_dtype, fp8_max
            )
            checks += _check_packed(
                rank, world_size, device, AttentionA2AIntraNodeOp, 32, packed_cases
            )
            checks += _check_hadamard(
                rank, world_size, device, AttentionA2AIntraNodeOp, fp8_dtype
            )
            checks += _check_reuse(
                rank, world_size, device, AttentionA2AIntraNodeOp, fp8_dtype, fp8_max
            )
            checks += _check_wan_2_2(
                rank, world_size, device, AttentionA2AIntraNodeOp, fp8_dtype, fp8_max
            )
        if run_tail:
            checks += _check_packed(
                rank,
                world_size,
                device,
                AttentionA2AIntraNodeOp,
                32,
                (_PackedCase("mxfp4-tail", "mxfp4", "mxfp4"),),
            )
            checks += _check_packed(
                rank,
                world_size,
                device,
                AttentionA2AIntraNodeOp,
                64,
                (
                    _PackedCase(
                        "mxfp6-mxfp4-fp6-p",
                        "mxfp6",
                        "mxfp4",
                        AttentionPack.V_FOR_FP6_P,
                    ),
                ),
            )
        return checks
    finally:
        try:
            ms.shmem_finalize()
        finally:
            dist.destroy_process_group()


def _run_world(world_size, run_full, run_tail, fp8_dtype, fp8_max, packed_cases):
    visible = torch.cuda.device_count()
    if not run_full and not run_tail:
        print(f"SKIP: world={world_size} has no selected suite", flush=True)
        return 0
    if visible < world_size:
        print(f"SKIP: requires {world_size} visible GPUs, found {visible}", flush=True)
        return 0
    port = _free_port()
    with get_context("spawn").Pool(processes=world_size) as pool:
        counts = pool.starmap(
            _run_rank,
            [
                (
                    rank,
                    world_size,
                    port,
                    run_full,
                    run_tail,
                    fp8_dtype,
                    fp8_max,
                    packed_cases,
                )
                for rank in range(world_size)
            ],
        )
    checks = sum(counts)
    print(f"PASS: world={world_size} checks={checks} failures=0", flush=True)
    return checks


def main():
    parser = argparse.ArgumentParser(description="Attention intranode A2A correctness")
    parser.add_argument(
        "--world-sizes",
        type=int,
        nargs="+",
        choices=(2, 4, 8),
        default=(8, 2),
        help="world sizes to run (default: 8 2)",
    )
    args = parser.parse_args()
    _check_scale_storage_extents()
    if not torch.cuda.is_available():
        print("SKIP: requires ROCm GPUs", flush=True)
        return 0
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in {"gfx942", "gfx950"}:
        print(f"SKIP: requires gfx942 or gfx950, attached GPU is {arch}", flush=True)
        return 0
    fp8_dtype, fp8_max = (
        (torch.float8_e4m3fnuz, 240.0)
        if arch == "gfx942"
        else (torch.float8_e4m3fn, 448.0)
    )
    packed_cases = _PACKED_CASES[:2] if arch == "gfx942" else _PACKED_CASES
    checks = 0
    for world_size in args.world_sizes:
        checks += _run_world(
            world_size,
            run_full=world_size == 8,
            run_tail=world_size == 2 and arch == "gfx950",
            fp8_dtype=fp8_dtype,
            fp8_max=fp8_max,
            packed_cases=packed_cases,
        )
    print(f"completed checks={checks} failures=0", flush=True)
    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
