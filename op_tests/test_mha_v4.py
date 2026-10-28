# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap

import pandas as pd
import torch
import torch._dynamo

import aiter
import aiter.ops.mha_v4 as mha_v4_module
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.mha_v4 import (
    AttentionFormat,
    AttentionPack,
    AttentionScaleMode,
    _k_mean,
    _RawRecipeKind,
    _resolve_raw_recipe,
    mha_v4,
    mha_v4_kv_tile,
    mha_v4_packed,
    native_fp8_format,
    scale_modes_for_formats,
)
from aiter.ops.mha_v4_quant import (
    MHA_V4_KV_SCALE_LOOKAHEAD_ROWS,
    MHA_V4_KV_TILE_ROWS,
    MHA_V4_LOG2E,
    MHA_V4_MXFP4_K_SCALE_SLACK_BYTES,
    MHA_V4_MXFP4_V_SCALE_SLACK_BYTES,
    MHA_V4_MXFP4_V_SCALE_TILE_BYTES,
    MHA_V4_MXFP6_V_BUFFER_SLACK_BYTES,
    mha_v4_q_multiplier,
    mxfp4_k_view,
    mxfp4_v_view,
    mxfp6_k_view,
    quantize_fp8,
    quantize_fp8_rotated,
    quantize_int8,
    quantize_mxfp4_k,
    quantize_mxfp4_q,
    quantize_mxfp6_k,
    quantize_mxfp8_k,
    quantize_mxfp8_q,
    quantize_v_fp8,
    quantize_v_mxfp4_fp6_p,
    quantize_v_mxfp6,
    quantize_v_mxfp6_fp6_p,
    rotate_activation_hd128,
    rotate_activation_mxfp6_quant,
)
from aiter.ops.triton.quant.mxfp6_fmha_pack import (
    _v_direct_kvtab,
    fp6_k_raw_buffer_sizes,
    quantize_fp6_v_clean_triton,
    quantize_fp6_v_data_scale_triton,
    reorder_fp6_k_lds_order_triton,
)
from aiter.ops.triton.quant.quant import dynamic_mxfp8_quant
from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
    fp4_v_padded_sequence,
    fp4_v_raw_buffer_size,
    pack_v_mxfp4_colmajor_raw,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

try:
    from mha_v4_check_harness import (
        CheckRegistry,
        expect_raises,
        param,
        parametrize,
        patched_attr,
        skipif,
    )
except ModuleNotFoundError as e:
    if e.name != "mha_v4_check_harness":
        raise
    from op_tests.mha_v4_check_harness import (
        CheckRegistry,
        expect_raises,
        param,
        parametrize,
        patched_attr,
        skipif,
    )


def _e2m1_code_ties_low(value):
    magnitude = value.abs()
    code = sum(
        magnitude > midpoint for midpoint in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    ).to(torch.uint8)
    return code | ((value < 0).to(torch.uint8) << 3)


def _rotate_hd128_reference(value):
    rotated = value.float()
    group_size = 1
    while group_size < 128:
        pairs = rotated.reshape(*value.shape[:-1], -1, 2, group_size)
        left = pairs[..., 0, :]
        right = pairs[..., 1, :]
        rotated = torch.cat((left + right, left - right), dim=-1).reshape(value.shape)
        group_size *= 2
    return (rotated / 128**0.5).to(value.dtype)


# Dynamo caches compiled entries per code object and this file compiles mha_v4 under many format
# combinations, so a shared cache drifts toward the recompile limit and whichever check compiles
# last fails under fullgraph. The kv-tile cache is cleared because patched_attr restores get_gfx
# but not a tile it may have warmed, and a stale tile is wrong mask geometry rather than an error.
_CHECKS = CheckRegistry(
    setup=torch._dynamo.reset,
    teardown=lambda: mha_v4_module.mha_v4_kv_tile.cache_clear(),
)


def test_attention_format_ids_are_stable():
    assert int(AttentionFormat.FP32) == 0
    assert int(AttentionFormat.FP16) == 1
    assert int(AttentionFormat.BF16) == 2
    assert int(AttentionFormat.FP8_E4M3) == 3
    assert AttentionFormat.FP8 is AttentionFormat.FP8_E4M3
    assert int(AttentionFormat.FP8_E4M3_FNUZ) == 4
    assert int(AttentionFormat.FP8_E5M2) == 5
    assert int(AttentionFormat.FP8_E5M2_FNUZ) == 6
    assert int(AttentionFormat.FP6_E2M3) == 7
    assert AttentionFormat.MXFP6 is AttentionFormat.FP6_E2M3
    assert int(AttentionFormat.FP6_E3M2) == 8
    assert AttentionFormat.MXBF6 is AttentionFormat.FP6_E3M2
    assert int(AttentionFormat.FP4_E2M1) == 9
    assert AttentionFormat.MXFP4 is AttentionFormat.FP4_E2M1
    assert int(AttentionFormat.INT8) == 10
    assert int(AttentionFormat.UINT8) == 11
    assert int(AttentionFormat.INT4) == 12
    assert int(AttentionFormat.UINT4) == 13
    assert int(AttentionPack.DEFAULT) == 0
    assert int(AttentionPack.V_FOR_FP6_P) == 1


def test_mha_v4_q_multiplier_recipe():
    softmax_scale = 128**-0.5
    assert mha_v4_q_multiplier(softmax_scale) == softmax_scale * MHA_V4_LOG2E


def test_mha_v4_f8f6_scale_recipe():
    assert scale_modes_for_formats(
        AttentionFormat.FP8, AttentionFormat.FP8, AttentionFormat.MXFP6
    ) == (
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.E8M0_PER_1X32,
    )


def test_mha_v4_bf16_scale_recipe():
    assert scale_modes_for_formats(
        AttentionFormat.BF16, AttentionFormat.BF16, AttentionFormat.BF16
    ) == (
        AttentionScaleMode.NONE,
        AttentionScaleMode.NONE,
        AttentionScaleMode.NONE,
    )


def test_mha_v4_bf16fp8_scale_recipe():
    assert scale_modes_for_formats(
        AttentionFormat.BF16, AttentionFormat.BF16, AttentionFormat.FP8
    ) == (
        AttentionScaleMode.NONE,
        AttentionScaleMode.NONE,
        AttentionScaleMode.F32_PER_TENSOR,
    )


@parametrize(
    ("q_format", "v_format", "sparse", "kind", "v_pack"),
    [
        (
            AttentionFormat.BF16,
            AttentionFormat.BF16,
            False,
            _RawRecipeKind.BF16,
            AttentionPack.DEFAULT,
        ),
        (
            AttentionFormat.FP8,
            AttentionFormat.MXFP6,
            False,
            _RawRecipeKind.FP8,
            AttentionPack.V_FOR_FP6_P,
        ),
        (
            AttentionFormat.FP8,
            AttentionFormat.MXFP6,
            True,
            _RawRecipeKind.FP8,
            AttentionPack.V_FOR_FP6_P,
        ),
        # All-MXFP4 consumes FP6 probabilities against MXFP4 V, so both modes need the repacked V.
        (
            AttentionFormat.MXFP4,
            AttentionFormat.MXFP4,
            False,
            _RawRecipeKind.MXFP4,
            AttentionPack.V_FOR_FP6_P,
        ),
        (
            AttentionFormat.MXFP4,
            AttentionFormat.MXFP4,
            True,
            _RawRecipeKind.MXFP4,
            AttentionPack.V_FOR_FP6_P,
        ),
        (
            AttentionFormat.MXFP6,
            AttentionFormat.MXFP4,
            False,
            _RawRecipeKind.MXFP6,
            AttentionPack.V_FOR_FP6_P,
        ),
        (
            AttentionFormat.MXFP6,
            AttentionFormat.MXFP4,
            True,
            _RawRecipeKind.MXFP6,
            AttentionPack.V_FOR_FP6_P,
        ),
        # MXFP6 Q/K/V ships an FP6-P object in both modes, so sparse keeps the repacked V.
        (
            AttentionFormat.MXFP6,
            AttentionFormat.MXFP6,
            True,
            _RawRecipeKind.MXFP6,
            AttentionPack.V_FOR_FP6_P,
        ),
    ],
)
def test_mha_v4_resolves_raw_recipe(q_format, v_format, sparse, kind, v_pack):
    recipe = _resolve_raw_recipe(
        q_format,
        q_format,
        v_format,
        None,
        None,
        None,
        sparse=sparse,
    )
    assert recipe.kind == kind
    assert recipe.v_pack == v_pack
    assert recipe.scale_modes == scale_modes_for_formats(q_format, q_format, v_format)


@parametrize(
    ("v_format", "kind"),
    [
        (
            AttentionFormat.BF16,
            _RawRecipeKind.BF16,
        ),
        (
            native_fp8_format(),
            _RawRecipeKind.BF16_FP8,
        ),
    ],
)
def test_mha_v4_resolves_bf16_sparse_recipe(v_format, kind):
    recipe = _resolve_raw_recipe(
        AttentionFormat.BF16,
        AttentionFormat.BF16,
        v_format,
        None,
        None,
        None,
        sparse=True,
    )
    assert recipe.kind == kind
    assert recipe.v_pack == AttentionPack.DEFAULT


@parametrize("sparse", [False, True])
def test_mha_v4_rejects_unimplemented_mxfp4_mxfp6_recipe(sparse):
    with expect_raises(NotImplementedError, match="MXFP4 Q/K requires MXFP4 V"):
        _resolve_raw_recipe(
            AttentionFormat.MXFP4,
            AttentionFormat.MXFP4,
            AttentionFormat.MXFP6,
            None,
            None,
            None,
            sparse=sparse,
        )


def test_mha_v4_rejects_f8f4_format_pair():
    with expect_raises(ValueError, match="matching FP8 or MXFP6 V"):
        scale_modes_for_formats(
            AttentionFormat.FP8, AttentionFormat.FP8, AttentionFormat.MXFP4
        )


def test_mha_v4_f8f6_v_kv_table_matches_live_p_pack():
    expected = []
    for lane in range(64):
        row = []
        for field in range(32):
            physical = 32 * (lane // 32) + field
            paired = (
                (physical & 0x0F) | ((physical & 0x10) << 1) | ((physical & 0x20) >> 1)
            )
            group, byte = divmod(paired, 32)
            row.append(
                32 * (byte // 16) + 8 * ((byte % 16) // 4) + byte % 4 + 4 * group
            )
        expected.append(row)

    assert _v_direct_kvtab().tolist() == expected


@skipif(get_gfx() != "gfx950", reason="gfx950 MXFP6 V packing")
def test_mha_v4_mxfp6_v_layout_contract():
    value = torch.randn((1, 256, 2, 128), device="cuda", dtype=torch.bfloat16)

    packed, scale = quantize_v_mxfp6(value)

    assert packed.shape == value.shape
    assert packed.dtype == torch.uint8
    assert packed.stride() == (2 * 2 * 12288, 96, 2 * 12288, 1)
    assert scale.shape == (1, 2, 2 * 512)
    assert scale.dtype == torch.uint8


@skipif(get_gfx() != "gfx950", reason="gfx950 MXFP6 V packing")
@parametrize("sequence", [256, 257])
@parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mha_v4_mxfp6_fp6_p_layout_matches_permuted_canonical(sequence, dtype):
    torch.manual_seed(41)
    value = torch.randn((1, sequence, 2, 128), device="cuda", dtype=dtype)

    canonical, canonical_scale = quantize_v_mxfp6(value)
    packed, scale = quantize_v_mxfp6_fp6_p(value)
    token = torch.arange(value.shape[1], device=value.device)
    within_block = token % 64
    paired = (
        (within_block & ~0x24)
        | ((within_block & 0x04) << 3)
        | ((within_block & 0x20) >> 3)
    )
    source_token = torch.minimum(
        token - within_block + paired, token.new_tensor(sequence - 1)
    )
    permuted = value[:, source_token].contiguous()
    expected, expected_scale = quantize_v_mxfp6(permuted)
    tiles = (sequence + 127) // 128
    data_size = value.shape[0] * value.shape[2] * tiles * 12288

    assert torch.equal(
        packed.as_strided((data_size,), (1,)),
        expected.as_strided((data_size,), (1,)),
    )
    assert torch.equal(scale.reshape(-1), expected_scale.reshape(-1))
    assert not torch.equal(packed, canonical)
    assert not torch.equal(scale, canonical_scale)

    # The producer zeroes trailing slack for the ASM's speculative reads, and comparing only
    # data_size bytes would let a regression there pass unnoticed.
    slack = MHA_V4_MXFP6_V_BUFFER_SLACK_BYTES
    assert packed.untyped_storage().nbytes() == data_size + slack
    assert torch.equal(
        packed.as_strided((slack,), (1,), data_size),
        torch.zeros(slack, device=packed.device, dtype=torch.uint8),
    )


@skipif(get_gfx() != "gfx950", reason="gfx950 F8F6 kernel")
def test_mha_v4_f8f6_raw_compile_parity():
    torch.manual_seed(43)
    q = torch.randn((1, 256, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 256, 2, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((1, 256, 2, 128), device="cuda", dtype=torch.bfloat16)
    eager_out = torch.empty_like(q)
    compiled_out = torch.empty_like(q)
    fp8_format = native_fp8_format()

    eager = mha_v4(
        q,
        k,
        v,
        fp8_format,
        fp8_format,
        AttentionFormat.MXFP6,
        out=eager_out,
    )
    compiled = torch.compile(mha_v4, fullgraph=True)(
        q,
        k,
        v,
        fp8_format,
        fp8_format,
        AttentionFormat.MXFP6,
        out=compiled_out,
    )
    torch.cuda.synchronize()

    assert eager.data_ptr() == eager_out.data_ptr()
    assert compiled.data_ptr() == compiled_out.data_ptr()
    assert torch.equal(compiled, eager)
    assert torch.isfinite(compiled).all()


@skipif(get_gfx() != "gfx950", reason="gfx950 MXFP6 V packing")
@parametrize("sequence", [256, 257])
def test_mha_v4_mxfp6_v_direct_buffers_match_combined_reference(sequence):
    torch.manual_seed(sequence)
    value = torch.randn((1, sequence, 2, 128), device="cuda", dtype=torch.bfloat16)
    tiles = (sequence + 127) // 128
    if sequence % 128:
        reference_value = torch.cat(
            [value, value[:, -1:].expand(-1, tiles * 128 - sequence, -1, -1)],
            dim=1,
        )
    else:
        reference_value = value

    combined = quantize_fp6_v_clean_triton(reference_value, direct_p=True).view(
        1, 2, tiles, 12800
    )
    data, scale = quantize_fp6_v_data_scale_triton(value)
    expected_data = combined[..., :12288].contiguous().view(-1)

    scale_tail = combined[..., 12288:].view(1, 2, tiles, 128, 4)
    lane = torch.arange(64, device=value.device)
    channel = lane[:, None] % 32 + 32 * torch.arange(4, device=value.device)[None, :]
    expected_scale = torch.stack(
        [scale_tail[..., channel, 2 * half + lane[:, None] // 32] for half in range(2)],
        dim=-3,
    ).contiguous()

    assert torch.equal(data[: expected_data.numel()], expected_data)
    assert torch.count_nonzero(data[expected_data.numel() :]) == 0
    assert torch.equal(scale, expected_scale.view(-1))


@skipif(get_gfx() != "gfx950", reason="gfx950 per-tensor quantization")
@parametrize("clip", [1.0, 0.9])
def test_mha_v4_int8_quantization_matches_torch(clip):
    torch.manual_seed(17)
    value = torch.randn((2, 257, 3, 128), device="cuda", dtype=torch.bfloat16)
    expected_scale = value.float().abs().max() * clip / 127.0
    expected = torch.clamp(torch.round(value.float() / expected_scale), -128, 127).to(
        torch.int8
    )

    actual, scale = quantize_int8(value, clip)

    assert torch.equal(actual, expected)
    assert torch.equal(scale, expected_scale.reshape(1))


@skipif(get_gfx() != "gfx950", reason="gfx950 per-tensor quantization")
def test_mha_v4_fp8_quantization_matches_torch():
    torch.manual_seed(19)
    value = torch.randn((2, 257, 3, 128), device="cuda", dtype=torch.bfloat16)
    expected_scale = value.float().abs().max() / torch.finfo(torch.float8_e4m3fn).max
    expected = (value.float() / expected_scale).to(torch.float8_e4m3fn)

    actual, scale = quantize_fp8(value)

    assert torch.equal(actual, expected)
    assert torch.equal(scale, expected_scale.reshape(1))


@parametrize("sequence,heads", [(1, 1), (257, 3), (1024, 2)])
def test_mha_v4_v_fp8_quantization_matches_torch(sequence, heads):
    torch.manual_seed(29)
    value = torch.randn((2, sequence, heads, 128), device="cuda", dtype=torch.bfloat16)
    fp8_max = torch.finfo(dtypes.fp8).max
    expected_scale = value.float().abs().amax(dim=1).clamp_min(1e-12) * (1.0 / fp8_max)

    actual, scale = quantize_v_fp8(value)

    assert torch.equal(scale, expected_scale)
    assert torch.equal(actual, (value.float() / scale[:, None]).to(dtypes.fp8))


@skipif(
    get_gfx() not in ("gfx942", "gfx950"),
    reason="gfx942/gfx950 hd128 rotation",
)
@parametrize("dtype", [torch.bfloat16, torch.float16])
@parametrize("sequence,heads", [(1, 1), (129, 5), (2048, 1)])
def test_mha_v4_rotation_matches_an_explicit_hadamard_matrix(dtype, sequence, heads):
    """The rotation is the orthonormal hd128 Walsh-Hadamard transform, not merely self-consistent.

    Checked against a Sylvester matrix applied as an fp32 matmul, which shares no code with the
    kernel's butterfly. That pins the transform and its 1/sqrt(128) normalization, where comparing
    against another kernel would only pin the two against each other -- and the recipe test below
    cannot help, because it evaluates the same rotation on both sides.

    The comparison allows one output ULP: the matmul reduces in a different order than the
    kernel's butterfly, and that only cancels when the 128 inputs share an exponent range.
    """
    torch.manual_seed(31)
    value = torch.randn((1, sequence, heads, 128), device="cuda", dtype=dtype)
    rotated = torch.empty_like(value)
    rotate_activation_hd128(rotated, value)

    index = torch.arange(128, device=value.device)
    overlap = index.view(-1, 1) & index.view(1, -1)
    parity = torch.zeros_like(overlap)
    for bit in range(7):
        parity ^= (overlap >> bit) & 1
    hadamard = torch.where(parity.bool(), -1.0, 1.0)
    assert torch.equal(
        hadamard @ hadamard.T / 128, torch.eye(128, device=value.device)
    ), "the reference matrix is not an orthonormal Hadamard matrix"

    expected = (value.float().reshape(-1, 128) @ hadamard) / math.sqrt(128)
    # One output ULP, not bitwise: an fp32 sum of 128 mixed-exponent 16-bit values is only exact
    # when they share an exponent range, so the matmul's order and the butterfly's can land a bit
    # apart. Demanding equality just pins the kernel to whatever order the installed torch picks.
    torch.testing.assert_close(
        rotated.float(),
        expected.to(dtype).reshape(value.shape).float(),
        rtol=torch.finfo(dtype).eps,
        atol=torch.finfo(dtype).tiny,
    )


@skipif(
    get_gfx() not in ("gfx942", "gfx950"),
    reason="gfx942/gfx950 rotated FP8 quantization",
)
@parametrize("sequence,heads", [(257, 3), (512, 1), (2048, 1)])
def test_mha_v4_rotated_fp8_quantization_matches_reference(sequence, heads):
    torch.manual_seed(23)
    value = torch.randn((1, heads, sequence, 128), device="cuda", dtype=torch.bfloat16)
    value = value.permute(0, 2, 1, 3).contiguous()
    expected_rotated = _rotate_hd128_reference(value)
    rotated = torch.empty_like(value)
    rotate_activation_hd128(rotated, value)
    expected, expected_scale = quantize_fp8(expected_rotated)

    actual, scale = quantize_fp8_rotated(value)

    assert torch.equal(rotated, expected_rotated)
    assert torch.equal(actual, expected)
    # Values in the message: a bare equality failure here is not reproducible off the arch that saw it.
    assert torch.equal(
        scale, expected_scale
    ), f"scale {scale.item():.9e} vs expected {expected_scale.item():.9e}"


def test_mha_v4_rotated_fp8_quantization_rejects_noncontiguous_input():
    value = torch.randn((1, 1, 128, 2), device="cuda", dtype=torch.bfloat16)
    value = value.transpose(-1, -2)

    with expect_raises(ValueError, match="requires contiguous hd128 input"):
        quantize_fp8_rotated(value)


# The MX quantizers AITER_CHECK on gfx950, and that check aborts rather than raising.
@skipif(get_gfx() != "gfx950", reason="gfx950 MX quantization")
@parametrize(
    "quantize",
    [
        "quantize_fp8_rotated",
        "quantize_mxfp8_k",
        "quantize_mxfp4_k",
        "quantize_mxfp6_k",
    ],
)
def test_mha_v4_k_quantizers_reject_an_off_device_mean(quantize):
    """A host mean reached the kernel as a device pointer and faulted the GPU.

    Out of process because these checks abort rather than raise, as every AITER_CHECK in that
    translation unit does. All four quantizers share one validator, so all four are covered.
    """
    source = textwrap.dedent(f"""
        import torch
        from aiter.ops.mha_v4_quant import {quantize} as quantize

        value = torch.randn((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)
        quantize(value, torch.zeros((1, 2, 128), dtype=torch.float32))
        torch.cuda.synchronize()
        """)
    finished = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )

    assert finished.returncode != 0
    assert "same GPU as input" in finished.stderr, finished.stderr[-2000:]


@skipif(
    get_gfx() not in ("gfx942", "gfx950"),
    reason="gfx942/gfx950 activation rotation",
)
def test_mha_v4_rotate_activation_hd128_accepts_empty_input():
    value = torch.empty((1, 0, 1, 128), device="cuda", dtype=torch.bfloat16)
    rotated = torch.empty_like(value)

    rotate_activation_hd128(rotated, value)

    assert rotated.shape == value.shape
    assert rotated.numel() == 0


# Runs out-of-process: the regression is a GPU memory fault, which aborts the interpreter rather
# than failing an assertion, and would take the rest of the session down with it.
_K_MEAN_OOB_PROBE = """
import os, sys
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
import torch
from aiter.ops.mha_v4_quant import (
    quantize_mxfp4_k,
    quantize_mxfp6_k,
    quantize_mxfp8_k,
    rotate_activation_hd128,
)

seq, heads = {seq}, {heads}
k = torch.randn(1, seq, heads, 128, device="cuda", dtype=torch.bfloat16)
# Park the mean flush against the end of its own allocation so an over-read faults instead of
# silently landing in a neighbour's memory.
buf = torch.zeros(2 * 1024 * 1024 // 4, device="cuda")
mean = buf[-heads * 128 :].view(1, heads, 128)

if "{kernel}" == "hd128":
    out = torch.empty_like(k)
    rotate_activation_hd128(out, k, mean)
else:
    {{"mxfp8": quantize_mxfp8_k, "mxfp6": quantize_mxfp6_k, "mxfp4": quantize_mxfp4_k}}[
        "{kernel}"
    ](k, mean)
torch.cuda.synchronize()
"""


_K_MEAN_OOB_SHAPES = [(3975, 3), (999, 3), (1001, 5)]
_K_MEAN_OOB_KERNELS = [
    "hd128",
    param("mxfp8", skip=get_gfx() != "gfx950"),
    param("mxfp6", skip=get_gfx() != "gfx950"),
    param("mxfp4", skip=get_gfx() != "gfx950"),
]


def check_k_mean_bound(kernel, seq, heads):
    """The rotate kernels work in blocks of 8 rows and subtract the K mean before the row < m
    guard, so a row count that is not a multiple of 8 used to read a mean row past the end.
    """
    assert (
        seq * heads
    ) % 8, "shape must leave a partial 8-row block to exercise the bound"
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            _K_MEAN_OOB_PROBE.format(seq=seq, heads=heads, kernel=kernel),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert probe.returncode == 0, (
        f"{kernel} faulted at seq={seq} heads={heads} "
        f"(rows={seq * heads}, rows%8={seq * heads % 8}): {probe.stderr[-400:]}"
    )


@skipif(
    get_gfx() not in ("gfx942", "gfx950"),
    reason="gfx942/gfx950 activation rotation",
)
@parametrize("kernel", _K_MEAN_OOB_KERNELS)
@parametrize("seq,heads", _K_MEAN_OOB_SHAPES)
def test_mha_v4_rotate_kernels_bound_the_k_mean_read(kernel, seq, heads):
    check_k_mean_bound(kernel, seq, heads)


@skipif(
    get_gfx() not in ("gfx942", "gfx950"),
    reason="gfx942/gfx950 FP8 recipe validation",
)
def test_mha_v4_fp8_raw_recipe_matches_rotated_packed():
    torch.manual_seed(29)
    q = torch.randn((1, 512, 5, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    fp8_format = native_fp8_format()

    # mha_v4 smooths K before quantizing; packed callers pass the same mean themselves.
    q_quantized, q_descale = quantize_fp8_rotated(q)
    k_quantized, k_descale = quantize_fp8_rotated(k, _k_mean(k, _RawRecipeKind.FP8))
    v_quantized, v_descale = quantize_fp8(v)
    expected = mha_v4_packed(
        q_quantized,
        k_quantized,
        v_quantized,
        q_descale,
        k_descale,
        v_descale,
        fp8_format,
        fp8_format,
        fp8_format,
        *scale_modes_for_formats(fp8_format, fp8_format, fp8_format),
    )

    actual = mha_v4(q, k, v, fp8_format, fp8_format, fp8_format)
    compiled = torch.compile(mha_v4, fullgraph=True)(
        q, k, v, fp8_format, fp8_format, fp8_format
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
    assert torch.equal(compiled, expected)


@skipif(
    get_gfx() not in ("gfx942", "gfx950"),
    reason="gfx942/gfx950 FP8 recipe validation",
)
# The FP8 arm is real gfx942 coverage; the MX arms would reach quantizers that AITER_CHECK on
# gfx950, and that check aborts the process rather than raising.
@parametrize(
    "recipe",
    [
        "fp8",
        param("mxfp8", skip=get_gfx() != "gfx950"),
        param("mxfp4", skip=get_gfx() != "gfx950"),
    ],
)
def test_mha_v4_quantized_tolerates_k_common_mode(recipe):
    """A direction shared by every key must not cost accuracy.

    Softmax is shift invariant in such a component, so the reference barely moves; only the
    quantizers care. Without the mean subtraction the error grows several-fold here, so this
    is the tripwire for silently dropping it.
    """
    torch.manual_seed(17)
    q = torch.randn((1, 1024, 4, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(q)
    base_k = torch.randn_like(q)
    direction = torch.randn((1, 1, 4, 128), device="cuda", dtype=torch.bfloat16)

    if recipe == "fp8":
        formats = (native_fp8_format(),) * 3
        scale_modes = {}
    elif recipe == "mxfp8":
        # MXFP8 is selected by the E8M0 scale modes, not by a distinct format.
        formats = (native_fp8_format(),) * 3
        scale_modes = {
            "q_scale_mode": AttentionScaleMode.E8M0_PER_1X32,
            "k_scale_mode": AttentionScaleMode.E8M0_PER_1X32,
            "v_scale_mode": AttentionScaleMode.F32_PER_TENSOR,
        }
    else:
        formats = (AttentionFormat.MXFP4,) * 3
        scale_modes = {}

    errors = []
    for common in (0.0, 16.0):
        k = base_k + common * direction
        reference = _dense_reference(q.float(), k.float(), v.float(), k.shape[1])
        out = mha_v4(q, k, v, *formats, **scale_modes)
        errors.append(
            ((out.float() - reference).norm() / reference.norm()).item(),
        )

    assert errors[1] < 1.5 * errors[0], (
        f"{recipe} degrades under a shared K direction: "
        f"{errors[0]:.4f} -> {errors[1]:.4f}; is K smoothing still applied?"
    )


# LSE itself is gfx950-only, so the gfx942 arm of this pair never had a path to pass.
@skipif(
    get_gfx() != "gfx950",
    reason="gfx950 LSE",
)
@parametrize("recipe", ["fp8", "mxfp8", "mxfp4"])
def test_mha_v4_lse_survives_k_common_mode(recipe):
    """K smoothing must not leak into the exported LSE.

    Smoothing runs the kernel against k - k_mean, shifting every score by the per-query constant
    q @ k_mean. Output cannot see it -- a shift shared by all keys cancels in the softmax -- so
    only the LSE carries it. Chunked consumers weight each chunk by exp(lse) and derive their own
    k_mean per chunk, so an uncorrected shift mis-weights the chunks; ring attention lost a third
    of its output norm this way while every output test stayed green.
    """
    torch.manual_seed(17)
    q = torch.randn((1, 1024, 4, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(q)
    base_k = torch.randn_like(q)
    direction = torch.randn((1, 1, 4, 128), device="cuda", dtype=torch.bfloat16)

    if recipe == "fp8":
        formats = (native_fp8_format(),) * 3
        scale_modes = {}
    elif recipe == "mxfp8":
        formats = (native_fp8_format(),) * 3
        scale_modes = {
            "q_scale_mode": AttentionScaleMode.E8M0_PER_1X32,
            "k_scale_mode": AttentionScaleMode.E8M0_PER_1X32,
            "v_scale_mode": AttentionScaleMode.F32_PER_TENSOR,
        }
    else:
        formats = (AttentionFormat.MXFP4,) * 3
        scale_modes = {}

    softmax_scale = q.shape[-1] ** -0.5
    errors = []
    for common in (0.0, 16.0):
        k = base_k + common * direction
        scores = (
            q.float().permute(0, 2, 1, 3) @ k.float().permute(0, 2, 3, 1)
        ) * softmax_scale
        reference = torch.logsumexp(scores, dim=-1)
        _, lse = mha_v4(
            q,
            k,
            v,
            *formats,
            return_lse=True,
            softmax_scale=softmax_scale,
            **scale_modes,
        )
        errors.append((lse.float() - reference).abs().max().item())

    assert (
        _k_mean(base_k + 16.0 * direction, _RawRecipeKind.FP8).abs().max() > 0
    ), "K smoothing did not engage, so this case cannot detect the leak"
    assert errors[1] < errors[0] + 0.5, (
        f"{recipe} LSE degrades under a shared K direction: "
        f"{errors[0]:.4f} -> {errors[1]:.4f} nats; is the k_mean shift still "
        "added back into the LSE?"
    )


@skipif(get_gfx() != "gfx950", reason="gfx950 MXFP8 quantization")
@parametrize("case", ["random", "zero", "powers", "extreme"])
def test_mha_v4_mxfp8_q_matches_unfused_pipeline(case):
    from aiter import dtypes

    if case == "random":
        torch.manual_seed(29)
        value = torch.randn((1, 129, 5, 128), device="cuda", dtype=torch.bfloat16)
    elif case == "zero":
        value = torch.zeros((1, 7, 5, 128), device="cuda", dtype=torch.bfloat16)
    elif case == "powers":
        powers = torch.tensor(
            [0.0] + [2.0**exponent for exponent in range(-12, 13)],
            device="cuda",
            dtype=torch.bfloat16,
        )
        value = powers.repeat((128 + powers.numel() - 1) // powers.numel())[:128]
        value = value.reshape(1, 1, 1, 128)
    else:
        # The butterfly grows intermediates 128x before its final divide, and bf16 shares f32's
        # exponent range, so bf16.max here rotates to inf/NaN and the two paths then only agree
        # on an encoding neither of them promises. This is the largest input that stays finite.
        value = torch.full(
            (1, 1, 1, 128),
            torch.finfo(torch.bfloat16).max / 128,
            device="cuda",
            dtype=torch.bfloat16,
        )

    multiplier = mha_v4_q_multiplier(128**-0.5)
    rotated = torch.empty_like(value)
    rotate_activation_hd128(rotated, value)
    expected, expected_scale = dynamic_mxfp8_quant(
        rotated * multiplier, quant_dtype=dtypes.fp8
    )

    actual, scale = quantize_mxfp8_q(value, multiplier)

    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(scale, expected_scale)


@skipif(get_gfx() != "gfx950", reason="gfx950 MXFP8 quantization")
@parametrize("case", ["random", "zero", "extreme"])
def test_mha_v4_mxfp8_k_matches_unfused_pipeline(case):
    from aiter import dtypes

    if case == "random":
        torch.manual_seed(41)
        value = torch.randn((1, 129, 5, 128), device="cuda", dtype=torch.bfloat16)
    elif case == "zero":
        value = torch.zeros((1, 7, 5, 128), device="cuda", dtype=torch.bfloat16)
    else:
        # Largest input the butterfly keeps finite; see the Q case above.
        value = torch.full(
            (1, 1, 1, 128),
            torch.finfo(torch.bfloat16).max / 128,
            device="cuda",
            dtype=torch.bfloat16,
        )

    rotated = torch.empty_like(value)
    rotate_activation_hd128(rotated, value)
    expected, expected_scale = dynamic_mxfp8_quant(rotated, quant_dtype=dtypes.fp8)

    actual, scale = quantize_mxfp8_k(value)

    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(scale, expected_scale)


@skipif(get_gfx() != "gfx950", reason="gfx950 per-tensor quantization")
@parametrize("quantize", [quantize_int8, quantize_fp8, quantize_fp8_rotated])
def test_mha_v4_per_tensor_quantization_handles_zero(quantize):
    value = torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)

    actual, scale = quantize(value)

    assert torch.count_nonzero(actual) == 0
    assert torch.equal(scale, torch.ones_like(scale))


def test_mha_v4_raw_buffer_sizes_are_stable():
    assert fp6_k_raw_buffer_sizes(1, 128, 1) == (17408 + 256, 128 * 4 + 64)
    assert fp6_k_raw_buffer_sizes(2, 129, 3) == (
        2 * 3 * 2 * 17408 + 256,
        2 * 129 * 3 * 4 + 64,
    )
    assert fp4_v_padded_sequence(128) == 128
    assert fp4_v_padded_sequence(129) == 256
    assert fp4_v_raw_buffer_size(2, 129, 3) == 2 * 256 * 3 * 64 + 64


@parametrize("batch,sequence,heads", [(1, 128, 5), (1, 129, 2), (2, 257, 3)])
def test_mha_v4_mxfp4_v_backing_storage_covers_logical_view(batch, sequence, heads):
    padded_sequence = fp4_v_padded_sequence(sequence)
    payload_size = batch * heads * padded_sequence * 64
    raw_size = fp4_v_raw_buffer_size(batch, sequence, heads)
    max_logical_offset = (
        (batch - 1) * heads * padded_sequence * 64
        + (sequence - 1) * 64
        + (heads - 1) * padded_sequence * 64
        + 127
    )

    assert raw_size == payload_size + 64
    assert max_logical_offset < raw_size


@skipif(get_gfx() != "gfx950", reason="gfx950 MXFP4 V validation")
@parametrize("sequence", [1, 63, 64, 127, 128, 129, 255, 257])
def test_mha_v4_mxfp4_fp6_p_pack_matches_permuted_canonical(sequence):
    torch.manual_seed(sequence)
    value = torch.randn((2, sequence, 3, 128), device="cuda", dtype=torch.bfloat16)
    token = torch.arange(sequence, device=value.device)
    within_block = token % 64
    paired = (
        (within_block & ~0x24)
        | ((within_block & 0x04) << 3)
        | ((within_block & 0x20) >> 3)
    )
    source_token = torch.minimum(
        token - within_block + paired, token.new_tensor(sequence - 1)
    )

    production_raw, production_scale = quantize_v_mxfp4_fp6_p(value)
    compiled_raw, compiled_scale = torch.compile(
        quantize_v_mxfp4_fp6_p, fullgraph=True
    )(value)
    expected_raw, expected_scale = pack_v_mxfp4_colmajor_raw(
        value[:, source_token].contiguous()
    )

    assert torch.equal(production_raw, expected_raw)
    assert torch.equal(production_scale, expected_scale)
    slack = MHA_V4_MXFP4_V_SCALE_SLACK_BYTES
    assert (
        production_scale.untyped_storage().nbytes() == production_scale.numel() + slack
    )
    assert torch.equal(
        production_scale.as_strided((slack,), (1,), production_scale.numel()),
        torch.zeros(slack, device="cuda", dtype=torch.uint8),
    )
    assert torch.equal(compiled_raw, expected_raw)
    assert torch.equal(compiled_scale, expected_scale)


@skipif(get_gfx() != "gfx950", reason="gfx950 MXFP6 K validation")
@parametrize("sequence", [128, 129, 257])
def test_mha_v4_mxfp6_k_raw_views(sequence):
    torch.manual_seed(sequence)
    value = torch.randn((2, sequence, 3, 128), device="cuda", dtype=torch.bfloat16)
    dense = torch.empty((2, sequence, 3, 96), device="cuda", dtype=torch.uint8)
    dense_scale = torch.empty((2, sequence, 3, 4), device="cuda", dtype=torch.uint8)
    rotate_activation_mxfp6_quant(dense, dense_scale, value, 1.0)
    expected_raw, expected_scale_raw = reorder_fp6_k_lds_order_triton(
        dense, dense_scale, return_raw=True
    )
    raw, scale_raw = quantize_mxfp6_k(value)
    packed, scale = mxfp6_k_view(raw, scale_raw, 2, sequence, 3)

    assert packed.shape == (2, sequence, 3, 96)
    assert scale.shape == (2, sequence, 3, 4)
    assert packed.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    assert scale.untyped_storage().data_ptr() == scale_raw.untyped_storage().data_ptr()
    tiles = (sequence + 127) // 128
    for batch_head in range(2 * 3):
        for tile in range(tiles):
            base = batch_head * tiles * 17408 + tile * 17408
            assert torch.equal(
                raw[base : base + 12288], expected_raw[base : base + 12288]
            )
            assert torch.equal(
                raw[base + 16384 : base + 17408],
                expected_raw[base + 16384 : base + 17408],
            )
    assert torch.equal(
        scale_raw[: dense_scale.numel()], expected_scale_raw[: dense_scale.numel()]
    )


@skipif(get_gfx() != "gfx950", reason="gfx950 MXFP4 K validation")
@parametrize("sequence", [1, 127, 128, 129, 257])
def test_mha_v4_mxfp4_k_coalesced_layout(sequence):
    torch.manual_seed(sequence)
    value = torch.randn((2, sequence, 3, 128), device="cuda", dtype=torch.bfloat16)
    dense, dense_scale = quantize_mxfp4_q(value, 1.0)
    raw, scale = quantize_mxfp4_k(value)
    coalesced = mxfp4_k_view(raw, scale)

    tiles = (sequence + 127) // 128
    token = torch.arange(sequence, device="cuda")
    chunk = torch.arange(4, device="cuda")
    byte = torch.arange(16, device="cuda")
    raw_offset = (
        torch.arange(2, device="cuda")[:, None, None, None, None] * (3 * tiles * 8192)
        + torch.arange(3, device="cuda")[None, None, :, None, None] * (tiles * 8192)
        + (token // 128)[None, :, None, None, None] * 8192
        + chunk[None, None, None, :, None] * 2048
        + (token % 128)[None, :, None, None, None] * 16
        + byte[None, None, None, None, :]
    )
    expected = dense.unflatten(-1, (4, 16))
    assert torch.equal(raw[raw_offset], expected)

    assert torch.equal(scale, dense_scale)
    # The gather addresses whole KV tiles plus the producer lookahead, so the backing storage has to
    # cover those rows and read as zero.
    padded = tiles * MHA_V4_KV_TILE_ROWS + MHA_V4_KV_SCALE_LOOKAHEAD_ROWS
    slack = (padded - sequence) * 3 * 4 + MHA_V4_MXFP4_K_SCALE_SLACK_BYTES
    assert scale.untyped_storage().nbytes() == scale.numel() + slack
    assert torch.equal(
        scale.as_strided((slack,), (1,), scale.numel()),
        torch.zeros(slack, device="cuda", dtype=torch.uint8),
    )
    assert coalesced.stride() == (3 * tiles * 8192, 64, tiles * 8192, 1)


def _llvm_objdump():
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    return shutil.which("llvm-objdump") or shutil.which(
        "llvm-objdump", path=f"{rocm}/llvm/bin"
    )


@skipif(_llvm_objdump() is None, reason="llvm-objdump not available")
@parametrize("arch", ["gfx942", "gfx950"])
def test_mha_v4_dispatchable_objects_bound_their_scale_gathers(arch):
    """No DISPATCHABLE fmha_v4 object may fetch a scale with an unguarded global load.

    This is the contract that lets the quant producers allocate Q scales at their logical
    extent: the gather is clamped by its buffer descriptor's num_records, so the query tile's
    overshoot past the sequence end reads zero instead of walking off the tensor. Nothing else
    can catch a regression -- the overshot rows are past seqlen_q, so their values never reach
    the output and every numeric test passes either way; the symptom is a fault at an unrelated
    later synchronization, and only when the over-read crosses an unmapped page.

    Driven off the manifest rather than the directory so that a row commented out there, whose
    object may predate the promotion, cannot be re-enabled without being rebuilt first.
    """
    asm_dir = os.environ.get(
        "AITER_ASM_DIR", os.path.join(mha_v4_module.AITER_ROOT_DIR, "hsa")
    )
    root = os.path.join(asm_dir, arch, "fmha_v4_fwd")
    with open(os.path.join(root, "fmha_v4_fwd.csv"), encoding="utf-8") as manifest:
        rows = [
            line.strip().split(",")
            for line in manifest
            if line.strip() and not line.startswith("#")
        ]
    header, rows = rows[0], rows[1:]
    co_names = sorted({row[header.index("co_name")] for row in rows})
    assert co_names, f"no dispatchable fmha_v4 rows for {arch}"

    # global_load_lds_dword is the bulk K-tile stage into LDS, not a scale fetch; it is bounded
    # by the FP4 K/V slack the producers still reserve.
    gather = re.compile(r"\bglobal_load_(?:dword|ubyte)\b")
    offenders = {}
    for name in co_names:
        # gfx942 names its objects with an MI300/ prefix; gfx950 does not.
        path = os.path.join(root, name)
        assert os.path.isfile(
            path
        ), f"{arch} manifest names {name}, which is not shipped"
        listing = subprocess.run(
            [_llvm_objdump(), "-d", f"--mcpu={arch}", path],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        hits = len(gather.findall(listing))
        if hits:
            offenders[name] = hits
    assert not offenders, (
        f"{arch} dispatchable objects fetch scales with unguarded global loads: {offenders}; "
        "the quant producers no longer reserve slack for an overshooting Q gather"
    )


@skipif(get_gfx() != "gfx950", reason="gfx950 scale gather validation")
@parametrize("sequence", [1, 128, 129, 257, 512])
@parametrize(
    "quantize",
    [quantize_v_mxfp4_fp6_p],
    ids=["fp6_p"],
)
def test_mha_v4_mxfp4_v_scale_backing_storage_covers_lookahead_tiles(
    quantize, sequence
):
    """The MXFP4 Q/K rows gather V scales two 512-byte tiles ahead of the running tile.

    The lead does not shrink at the end of the sequence, so the last tiles address scale bytes
    past the final one whatever the length. Measured on gfx950: 1023 trailing mapped bytes still
    fault, 1024 do not.
    """
    heads = 3
    value = torch.randn((2, sequence, heads, 128), device="cuda", dtype=torch.bfloat16)
    _, scale = quantize(value)

    slack = MHA_V4_MXFP4_V_SCALE_SLACK_BYTES
    assert slack == 2 * MHA_V4_MXFP4_V_SCALE_TILE_BYTES
    assert scale.is_contiguous()
    assert scale.untyped_storage().nbytes() == scale.numel() + slack
    assert torch.equal(
        scale.as_strided((slack,), (1,), scale.numel()),
        torch.zeros(slack, device="cuda", dtype=torch.uint8),
    )


# _check_lse_capable rejects every non-gfx950 arch before the recipe is looked at.
@skipif(get_gfx() != "gfx950", reason="gfx950 LSE")
def test_mha_v4_rejects_unsupported_contracts():
    q = torch.empty((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)
    # LSE ships for every measured format row, dense and sorted-sparse alike; a row outside that
    # set is rejected rather than left to export a value nothing has compared against. Off gfx950
    # the architecture gate fires first, so that is the message to expect there.
    block_mask = torch.ones((1, 2, 1, 1), device="cuda", dtype=torch.bool)
    lse_rejection = (
        "not implemented for Q=FP4_E2M1"
        if get_gfx() == "gfx950"
        else "LSE is not validated on"
    )
    with expect_raises(NotImplementedError, match=lse_rejection):
        mha_v4(
            q,
            q,
            q,
            AttentionFormat.MXFP4,
            AttentionFormat.MXFP4,
            AttentionFormat.FP8,
            return_lse=True,
            block_mask=block_mask,
        )
    with expect_raises(ValueError, match="matching Q and K formats"):
        mha_v4(
            q,
            q,
            q,
            AttentionFormat.FP8,
            AttentionFormat.INT8,
            AttentionFormat.FP8,
        )


@skipif(get_gfx() != "gfx950", reason="gfx950 MHA v4 validation")
@parametrize(
    ("q_format", "v_format", "scale_modes"),
    [
        (AttentionFormat.BF16, AttentionFormat.BF16, None),
        (AttentionFormat.BF16, AttentionFormat.FP8, None),
        (AttentionFormat.INT8, AttentionFormat.FP8, None),
        (AttentionFormat.FP8, AttentionFormat.FP8, None),
        (AttentionFormat.FP8, AttentionFormat.MXFP6, None),
        (AttentionFormat.MXFP4, AttentionFormat.MXFP4, None),
        (AttentionFormat.MXFP6_E2M3, AttentionFormat.FP8, None),
        (AttentionFormat.MXFP6_E2M3, AttentionFormat.MXFP6, None),
        (AttentionFormat.MXFP6_E2M3, AttentionFormat.MXFP4, None),
        (
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            (
                AttentionScaleMode.E8M0_PER_1X32,
                AttentionScaleMode.E8M0_PER_1X32,
                AttentionScaleMode.F32_PER_TENSOR,
            ),
        ),
    ],
)
def test_mha_v4_dense_lse_matches_reference(q_format, v_format, scale_modes):
    """A wrong LSE does not fail output validation, so it needs its own reference check.

    The quantized rows carry a small systematic bias from the approximate exp2; the bound here is
    wide enough to pass that but far tighter than the failure modes it guards, which are a missing
    log2(L) term (error grows like ln(Sk)) and an unapplied P-pack divisor (a constant ln2 or 2ln2).
    """
    torch.manual_seed(31)
    q = torch.randn((1, 512, 5, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    softmax_scale = 128**-0.5
    qsm, ksm, vsm = scale_modes if scale_modes else (None, None, None)
    kwargs = {
        "softmax_scale": softmax_scale,
        "q_scale_mode": qsm,
        "k_scale_mode": ksm,
        "v_scale_mode": vsm,
    }

    out_only = mha_v4(q, k, v, q_format, q_format, v_format, **kwargs)
    out, lse = mha_v4(q, k, v, q_format, q_format, v_format, return_lse=True, **kwargs)

    scores = torch.matmul(
        q.float().permute(0, 2, 1, 3),
        k.float().permute(0, 2, 1, 3).transpose(-1, -2),
    )
    reference = torch.logsumexp(scores * softmax_scale, dim=-1)

    assert lse.shape == reference.shape
    assert lse.dtype == torch.float32
    assert torch.isfinite(lse).all()
    # Asking for the LSE must not perturb O: the epilogue runs either way and only the store is
    # gated, so its scratch registers must not touch anything O still needs.
    assert torch.equal(out_only, out)
    error = (lse - reference).abs()
    assert error.max().item() < 0.25, error.max().item()
    assert error.mean().item() < 0.05, error.mean().item()


@skipif(get_gfx() != "gfx950", reason="gfx950 MHA v4 validation")
@parametrize(
    ("q_format", "v_format"),
    [
        (AttentionFormat.BF16, AttentionFormat.BF16),
        (AttentionFormat.BF16, AttentionFormat.FP8),
        (AttentionFormat.INT8, AttentionFormat.FP8),
        (AttentionFormat.FP8, AttentionFormat.FP8),
        (AttentionFormat.MXFP6_E2M3, AttentionFormat.MXFP6),
        (AttentionFormat.MXFP4, AttentionFormat.MXFP4),
    ],
)
@parametrize("gqa_ratio", [2, 4])
def test_mha_v4_dense_lse_under_gqa(q_format, v_format, gqa_ratio):
    """K smoothing holds one mean per KV head, so the LSE correction has to expand it.

    Every quantized recipe derives a k_mean; contracting it against Q without expanding across
    the query-head group is a shape error, which is why this needs its own case per recipe
    rather than one BF16 smoke test. BF16 rows carry no k_mean and exercise only the plumbing.
    """
    torch.manual_seed(37)
    heads_k = 2
    heads_q = heads_k * gqa_ratio
    dim = 128
    q = torch.randn((1, 256, heads_q, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 256, heads_k, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    softmax_scale = dim**-0.5

    _, lse = mha_v4(
        q,
        k,
        v,
        q_format,
        q_format,
        v_format,
        softmax_scale=softmax_scale,
        return_lse=True,
    )

    # Expand the way the kernel does: query head j reads KV head j // ratio.
    expanded = k.repeat_interleave(gqa_ratio, dim=2)
    scores = torch.matmul(
        q.float().permute(0, 2, 1, 3),
        expanded.float().permute(0, 2, 1, 3).transpose(-1, -2),
    )
    reference = torch.logsumexp(scores * softmax_scale, dim=-1)

    assert lse.shape == reference.shape
    assert torch.isfinite(lse).all()
    error = (lse - reference).abs().max().item()
    assert error < 0.25, error


@skipif(get_gfx() != "gfx950", reason="gfx950 LSE")
@parametrize(
    ("q_format", "v_format", "max_spread"),
    [
        (AttentionFormat.BF16, AttentionFormat.BF16, 0.05),
        (AttentionFormat.BF16, AttentionFormat.FP8, 0.05),
        (AttentionFormat.INT8, AttentionFormat.FP8, 0.05),
        (AttentionFormat.FP8, AttentionFormat.FP8, 0.75),
        (AttentionFormat.FP8, AttentionFormat.MXFP6, 0.75),
        (AttentionFormat.MXFP4, AttentionFormat.MXFP4, 0.35),
        (AttentionFormat.MXFP6_E2M3, AttentionFormat.FP8, 0.35),
        (AttentionFormat.MXFP6_E2M3, AttentionFormat.MXFP6, 0.35),
        (AttentionFormat.MXFP6_E2M3, AttentionFormat.MXFP4, 0.35),
    ],
)
def test_mha_v4_lse_bias_is_constant_across_key_chunks(q_format, v_format, max_spread):
    """Ring weights each chunk by exp(lse), so only a bias identical across chunks cancels.

    The absolute bias is allowed to be nonzero and the test above already bounds it; what this
    one pins is that it does not move from chunk to chunk. Keys escalate along the sequence on
    purpose: random ones keep every row diffuse, the frozen-max conversion gate never trips, and
    a rollback missing from the exported max is then invisible. With this input the recipes span
    0.00 to 0.24 nats, while that defect measured 4.55.
    """
    torch.manual_seed(31)
    batch, sequence, heads, head_dim, chunks = 1, 2048, 5, 128, 4
    query = torch.randn(
        (batch, sequence, heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    value = torch.randn_like(query)
    ramp = torch.linspace(1.0, 6.0, sequence, device="cuda", dtype=torch.float32)
    key = (torch.randn_like(query).float() * ramp.view(1, -1, 1, 1)).to(torch.bfloat16)
    softmax_scale = head_dim**-0.5

    span = sequence // chunks
    biases = []
    for start in range(0, sequence, span):
        key_chunk = key[:, start : start + span]
        _, lse = mha_v4(
            query,
            key_chunk,
            value[:, start : start + span],
            q_format,
            q_format,
            v_format,
            softmax_scale=softmax_scale,
            return_lse=True,
        )
        scores = query.float().permute(0, 2, 1, 3) @ key_chunk.float().permute(
            0, 2, 3, 1
        )
        reference = torch.logsumexp(scores * softmax_scale, dim=-1)
        biases.append((lse.float() - reference).mean().item())

    spread = max(biases) - min(biases)
    assert spread < max_spread, f"per-chunk bias {biases} spans {spread:.4f} nats"


@skipif(get_gfx() != "gfx950", reason="gfx950 LSE")
def test_mha_v4_lse_is_gated_off_gfx950():
    """gfx942 carries the epilogue, but its exported value has never been measured.

    A wrong LSE passes every output test, because O never reads it, so presence of the store is
    not evidence of correctness. Drop the gate once MI300 is compared against torch.logsumexp.
    """
    with patched_attr(mha_v4_module, "get_gfx", lambda: "gfx942"):
        q = torch.randn((1, 128, 4, 128), device="cuda", dtype=torch.bfloat16)
        formats = (AttentionFormat.BF16, AttentionFormat.BF16, AttentionFormat.BF16)

        with expect_raises(NotImplementedError, match="not validated on gfx942"):
            mha_v4(q, q, q, *formats, return_lse=True)

        assert torch.isfinite(mha_v4(q, q, q, *formats)).all()


@parametrize(
    "q_format",
    [
        AttentionFormat.FP16,
        AttentionFormat.FP8_E5M2,
        AttentionFormat.FP8_E5M2_FNUZ,
        AttentionFormat.UINT8,
        AttentionFormat.INT4,
        AttentionFormat.UINT4,
    ],
)
def test_mha_v4_rejects_reserved_raw_formats(q_format):
    q = torch.empty((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)
    with expect_raises((ValueError, NotImplementedError)):
        mha_v4(
            q,
            q,
            q,
            q_format,
            q_format,
            AttentionFormat.FP8,
        )


def test_mha_v4_raw_rejects_partial_scale_recipe():
    q = torch.empty((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)
    with expect_raises(ValueError, match="must all be set or all omitted"):
        mha_v4(
            q,
            q,
            q,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            q_scale_mode=AttentionScaleMode.E8M0_PER_1X32,
        )


def test_mha_v4_raw_rejects_unsupported_scale_recipe():
    q = torch.empty((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)
    with expect_raises(ValueError, match="unsupported scale recipe"):
        mha_v4(
            q,
            q,
            q,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            q_scale_mode=AttentionScaleMode.E8M0_PER_1X32,
            k_scale_mode=AttentionScaleMode.E8M0_PER_1X32,
            v_scale_mode=AttentionScaleMode.F32_PER_CHANNEL,
        )


@skipif(get_gfx() != "gfx950", reason="gfx950 MHA v4 validation")
def test_mha_v4_packed_rejects_wrong_scale_recipe():
    q = torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.int8)
    v = torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    with expect_raises(ValueError, match="unsupported scale recipe"):
        mha_v4_packed(
            q,
            q,
            v,
            scale,
            scale,
            scale,
            AttentionFormat.INT8,
            AttentionFormat.INT8,
            AttentionFormat.FP8,
            AttentionScaleMode.E8M0_PER_1X32,
            AttentionScaleMode.E8M0_PER_1X32,
            AttentionScaleMode.F32_PER_CHANNEL,
        )


@skipif(get_gfx() != "gfx950", reason="gfx950 MXFP8 validation")
def test_mha_v4_packed_accepts_mxfp8_scale_recipe():
    q = torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.float8_e4m3fn)
    qk_scale = torch.ones((1, 128, 2, 4), device="cuda", dtype=torch.uint8)
    v_scale = torch.ones(1, device="cuda", dtype=torch.float32)
    mha_v4_packed(
        q,
        q,
        q,
        qk_scale,
        qk_scale,
        v_scale,
        AttentionFormat.FP8,
        AttentionFormat.FP8,
        AttentionFormat.FP8,
        AttentionScaleMode.E8M0_PER_1X32,
        AttentionScaleMode.E8M0_PER_1X32,
        AttentionScaleMode.F32_PER_TENSOR,
    )


@skipif(get_gfx() != "gfx950", reason="gfx950 MHA v4 validation")
def test_mha_v4_packed_rejects_wrong_fp8_encoding():
    q = torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    with expect_raises(RuntimeError, match="must be FP8 E4M3 FNUZ"):
        mha_v4_packed(
            q,
            q,
            q,
            scale,
            scale,
            scale,
            AttentionFormat.FP8_E4M3_FNUZ,
            AttentionFormat.FP8_E4M3_FNUZ,
            AttentionFormat.FP8_E4M3_FNUZ,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
        )


@skipif(get_gfx() != "gfx950", reason="gfx950 MXFP4 K validation")
def test_mha_v4_packed_rejects_wrong_mxfp4_k_layout():
    q = torch.zeros((1, 128, 2, 64), device="cuda", dtype=torch.uint8)
    scale = torch.ones((1, 128, 2, 4), device="cuda", dtype=torch.uint8)
    v_fp8 = torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.float8_e4m3fn)
    v_scale = torch.ones((1, 2, 128), device="cuda", dtype=torch.float32)

    for v_format, value, value_scale, v_scale_mode in (
        (AttentionFormat.FP8, v_fp8, v_scale, AttentionScaleMode.F32_PER_CHANNEL),
        (
            AttentionFormat.MXFP4,
            q.new_zeros((1, 128, 2, 128)),
            q.new_zeros((1, 2, 512)),
            AttentionScaleMode.E8M0_PER_1X32,
        ),
    ):
        with expect_raises(ValueError, match="coalesced MHA v4 tile layout"):
            mha_v4_packed(
                q,
                q,
                value,
                scale,
                scale,
                value_scale,
                AttentionFormat.MXFP4,
                AttentionFormat.MXFP4,
                v_format,
                AttentionScaleMode.E8M0_PER_1X32,
                AttentionScaleMode.E8M0_PER_1X32,
                v_scale_mode,
            )

    raw, k_scale = quantize_mxfp4_k(
        torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)
    )
    coalesced_k = mxfp4_k_view(raw, k_scale)
    assert coalesced_k.stride() == (16384, 64, 8192, 1)


@skipif(get_gfx() != "gfx950", reason="gfx950 MX scale validation")
def test_mha_v4_packed_rejects_unbacked_mx_scales():
    """An external caller passing exact-size scales would fault the speculative gathers.

    clone() keeps the logical shape the size checks look at but drops the producer's
    zeroed slack, which is exactly the shape of a caller-supplied tensor. Q is not checked:
    its gather is bounded by a buffer descriptor, so it needs no slack.
    """
    torch.manual_seed(5)
    value = torch.randn((1, 257, 2, 128), device="cuda", dtype=torch.bfloat16)

    mxfp4_q, mxfp4_q_scale = quantize_mxfp4_q(value, 1.0)
    mxfp4_raw, mxfp4_k_scale = quantize_mxfp4_k(value)
    mxfp4_v_raw, mxfp4_v_scale = quantize_v_mxfp4_fp6_p(value)
    with expect_raises(RuntimeError, match="speculative tile gather"):
        mha_v4_packed(
            mxfp4_q,
            mxfp4_k_view(mxfp4_raw, mxfp4_k_scale),
            mxfp4_v_view(mxfp4_v_raw, mxfp4_v_scale, value.shape[1]),
            mxfp4_q_scale,
            mxfp4_k_scale.clone(),
            mxfp4_v_scale,
            AttentionFormat.MXFP4,
            AttentionFormat.MXFP4,
            AttentionFormat.MXFP4,
            AttentionScaleMode.E8M0_PER_1X32,
            AttentionScaleMode.E8M0_PER_1X32,
            AttentionScaleMode.E8M0_PER_1X32,
        )

    with expect_raises(RuntimeError, match="MX V descale needs"):
        mha_v4_packed(
            mxfp4_q,
            mxfp4_k_view(mxfp4_raw, mxfp4_k_scale),
            mxfp4_v_view(mxfp4_v_raw, mxfp4_v_scale, value.shape[1]),
            mxfp4_q_scale,
            mxfp4_k_scale,
            mxfp4_v_scale.clone(),
            AttentionFormat.MXFP4,
            AttentionFormat.MXFP4,
            AttentionFormat.MXFP4,
            AttentionScaleMode.E8M0_PER_1X32,
            AttentionScaleMode.E8M0_PER_1X32,
            AttentionScaleMode.E8M0_PER_1X32,
        )


@skipif(get_gfx() != "gfx950", reason="gfx950 MHA v4 validation")
@parametrize(
    ("q_format", "v_format"),
    [
        (AttentionFormat.BF16, AttentionFormat.BF16),
        (AttentionFormat.BF16, AttentionFormat.FP8),
        (AttentionFormat.INT8, AttentionFormat.FP8),
        (AttentionFormat.FP8, AttentionFormat.FP8),
    ],
)
def test_mha_v4_zero_inputs_are_finite(q_format, v_format):
    q = torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.bfloat16)
    out = mha_v4(q, q, q, q_format, q_format, v_format)
    torch.cuda.synchronize()
    assert torch.count_nonzero(out) == 0
    assert torch.isfinite(out).all()


@skipif(get_gfx() != "gfx950", reason="gfx950 MHA v4 validation")
@parametrize(
    ("q_format", "v_format", "scale_modes"),
    [
        (AttentionFormat.BF16, AttentionFormat.BF16, {}),
        (AttentionFormat.BF16, AttentionFormat.FP8, {}),
        (AttentionFormat.INT8, AttentionFormat.FP8, {}),
        (AttentionFormat.FP8, AttentionFormat.FP8, {}),
        (
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            {
                "q_scale_mode": AttentionScaleMode.E8M0_PER_1X32,
                "k_scale_mode": AttentionScaleMode.E8M0_PER_1X32,
                "v_scale_mode": AttentionScaleMode.F32_PER_TENSOR,
            },
        ),
        (AttentionFormat.FP8, AttentionFormat.MXFP6, {}),
        (AttentionFormat.MXFP6, AttentionFormat.FP8, {}),
        (AttentionFormat.MXFP6, AttentionFormat.MXFP6, {}),
        (AttentionFormat.MXFP6, AttentionFormat.MXFP4, {}),
        (AttentionFormat.MXFP4, AttentionFormat.MXFP4, {}),
    ],
)
def test_mha_v4_empty_heads_are_finite(q_format, v_format, scale_modes):
    """Sequence-parallel head padding leaves whole (batch, head) slices zero.

    Only reachable with live heads alongside them: f6f8 returned NaN for exactly half its output
    here, because its per-channel FP8 V quantizer divided by a zero amax while the populated heads
    kept the tensor looking healthy.
    """
    torch.manual_seed(0)
    q = torch.randn((1, 512, 4, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    for tensor in (q, k, v):
        tensor[:, :, 2:] = 0

    out = mha_v4(q, k, v, q_format, q_format, v_format, **scale_modes)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all(), (
        f"{q_format.name}/{v_format.name} produced "
        f"{int(torch.isnan(out).sum())} NaN on padded heads"
    )
    assert torch.count_nonzero(out[:, :, 2:]) == 0
    assert torch.count_nonzero(out[:, :, :2]) > 0


@skipif(get_gfx() != "gfx950", reason="gfx950 BF16-FP8 validation")
@parametrize(("sequence_q", "sequence_k"), [(129, 257), (257, 193)])
def test_mha_v4_bf16fp8_matches_dequantized_reference(sequence_q, sequence_k):
    torch.manual_seed(sequence_q + sequence_k)
    q = torch.randn((1, sequence_q, 5, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, sequence_k, 5, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    v_quantized, v_descale = quantize_fp8(v)
    v_dequantized = v_quantized.float() * v_descale
    scores = torch.matmul(
        q.transpose(1, 2).float(), k.transpose(1, 2).float().transpose(-1, -2)
    ) * (128**-0.5)
    reference = torch.matmul(
        torch.softmax(scores, dim=-1), v_dequantized.transpose(1, 2)
    ).transpose(1, 2)

    actual = mha_v4(
        q,
        k,
        v,
        AttentionFormat.BF16,
        AttentionFormat.BF16,
        AttentionFormat.FP8,
    )
    torch.cuda.synchronize()

    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), reference.flatten(), dim=0
    )
    assert torch.isfinite(actual).all()
    assert cosine > 0.998


@skipif(get_gfx() != "gfx950", reason="gfx950 GQA validation")
def test_mha_v4_mxfp4_gqa_matches_repeated_kv():
    torch.manual_seed(41)
    q = torch.randn((2, 129, 64, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((2, 257, 4, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    gqa = mha_v4(
        q,
        k,
        v,
        AttentionFormat.MXFP4,
        AttentionFormat.MXFP4,
        AttentionFormat.MXFP4,
    )
    mha = mha_v4(
        q,
        k.repeat_interleave(16, dim=2),
        v.repeat_interleave(16, dim=2),
        AttentionFormat.MXFP4,
        AttentionFormat.MXFP4,
        AttentionFormat.MXFP4,
    )
    torch.cuda.synchronize()

    assert torch.equal(gqa, mha)


@skipif(get_gfx() not in ("gfx942", "gfx950"), reason="gfx942/gfx950 I8FP8 validation")
def test_mha_v4_packed_i8fp8_compile_parity():
    torch.manual_seed(17)
    q = torch.randint(-32, 33, (1, 512, 5, 128), device="cuda", dtype=torch.int8)
    k = torch.randint(-32, 33, (1, 512, 5, 128), device="cuda", dtype=torch.int8)
    v = torch.randn((1, 512, 5, 128), device="cuda").to(dtypes.fp8)
    q_descale = torch.tensor([0.02], device="cuda")
    k_descale = torch.tensor([0.03], device="cuda")
    v_descale = torch.tensor([0.04], device="cuda")
    scale = 128**-0.5
    fp8_format = native_fp8_format()

    eager = mha_v4_packed(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        AttentionFormat.INT8,
        AttentionFormat.INT8,
        fp8_format,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        softmax_scale=scale,
    )
    compiled = torch.compile(mha_v4_packed, fullgraph=True)(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        AttentionFormat.INT8,
        AttentionFormat.INT8,
        fp8_format,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        softmax_scale=scale,
    )
    torch.cuda.synchronize()
    assert torch.equal(eager, compiled)


@skipif(get_gfx() not in ("gfx942", "gfx950"), reason="gfx942/gfx950 FP8 validation")
def test_mha_v4_packed_fp8_compile_parity():
    torch.manual_seed(23)
    q = torch.randn((1, 512, 5, 128), device="cuda").to(dtypes.fp8)
    k = torch.randn((1, 512, 5, 128), device="cuda").to(dtypes.fp8)
    v = torch.randn((1, 512, 5, 128), device="cuda").to(dtypes.fp8)
    q_descale = torch.tensor([0.02], device="cuda")
    k_descale = torch.tensor([0.03], device="cuda")
    v_descale = torch.tensor([0.04], device="cuda")
    fp8_format = native_fp8_format()
    scale_modes = scale_modes_for_formats(fp8_format, fp8_format, fp8_format)
    scale = 128**-0.5

    eager = mha_v4_packed(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        fp8_format,
        fp8_format,
        fp8_format,
        *scale_modes,
        softmax_scale=scale,
    )
    compiled = torch.compile(mha_v4_packed, fullgraph=True)(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        fp8_format,
        fp8_format,
        fp8_format,
        *scale_modes,
        softmax_scale=scale,
    )
    torch.cuda.synchronize()
    assert torch.equal(eager, compiled)


@skipif(get_gfx() != "gfx950", reason="gfx950 MHA v4 validation")
def test_mha_v4_native_schema_mutates_only_out():
    q = torch.zeros((1, 128, 2, 128), device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    mha_v4_packed(
        q,
        q,
        q,
        scale,
        scale,
        scale,
        AttentionFormat.FP8,
        AttentionFormat.FP8,
        AttentionFormat.FP8,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
    )

    schema = str(torch.ops.aiter.mha_v4_fwd_launch.default._schema)
    assert "Tensor q" in schema
    assert "Tensor k" in schema
    assert "Tensor v" in schema
    assert "Tensor(a6!) out" in schema
    assert schema.endswith("-> ()")


@skipif(get_gfx() != "gfx950", reason="gfx950 MHA v4 validation")
@parametrize(
    ("q_format", "v_format"),
    [
        (AttentionFormat.BF16, AttentionFormat.BF16),
        (AttentionFormat.BF16, AttentionFormat.FP8),
        (AttentionFormat.INT8, AttentionFormat.FP8),
        (AttentionFormat.FP8, AttentionFormat.FP8),
        (AttentionFormat.FP8, AttentionFormat.MXFP6),
        (AttentionFormat.MXFP4, AttentionFormat.MXFP4),
        (AttentionFormat.MXFP6_E2M3, AttentionFormat.FP8),
        (AttentionFormat.MXFP6_E2M3, AttentionFormat.MXFP6),
        (AttentionFormat.MXFP6_E2M3, AttentionFormat.MXFP4),
    ],
)
def test_mha_v4_raw_compile_parity(q_format, v_format):
    torch._dynamo.reset()
    torch.manual_seed(31)
    q = torch.randn((1, 512, 5, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    eager_out = torch.empty_like(q)
    compiled_out = torch.empty_like(q)

    eager = mha_v4(q, k, v, q_format, q_format, v_format, out=eager_out)
    compiled = torch.compile(mha_v4, fullgraph=True)(
        q, k, v, q_format, q_format, v_format, out=compiled_out
    )
    churn = torch.empty((16 * 1024 * 1024,), device="cuda", dtype=torch.uint8)
    consumed = compiled.contiguous()
    torch.cuda.synchronize()

    assert eager.data_ptr() == eager_out.data_ptr()
    assert compiled.data_ptr() == compiled_out.data_ptr()
    assert torch.equal(eager, compiled)
    assert torch.isfinite(consumed).all()
    assert churn.numel() == 16 * 1024 * 1024


@skipif(get_gfx() != "gfx950", reason="gfx950 MXFP8 validation")
def test_mha_v4_raw_mxfp8_compile_parity():
    torch.manual_seed(41)
    q = torch.randn((1, 257, 5, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    eager_out = torch.empty_like(q)
    compiled_out = torch.empty_like(q)

    fp8_format = native_fp8_format()
    scale_modes = (
        AttentionScaleMode.E8M0_PER_1X32,
        AttentionScaleMode.E8M0_PER_1X32,
        AttentionScaleMode.F32_PER_TENSOR,
    )
    eager = mha_v4(
        q,
        k,
        v,
        fp8_format,
        fp8_format,
        fp8_format,
        out=eager_out,
        q_scale_mode=scale_modes[0],
        k_scale_mode=scale_modes[1],
        v_scale_mode=scale_modes[2],
    )
    compiled = torch.compile(mha_v4, fullgraph=True)(
        q,
        k,
        v,
        fp8_format,
        fp8_format,
        fp8_format,
        out=compiled_out,
        q_scale_mode=scale_modes[0],
        k_scale_mode=scale_modes[1],
        v_scale_mode=scale_modes[2],
    )
    torch.cuda.synchronize()

    assert eager.data_ptr() == eager_out.data_ptr()
    assert compiled.data_ptr() == compiled_out.data_ptr()
    assert torch.equal(eager, compiled)
    assert torch.isfinite(compiled).all()


@skipif(get_gfx() != "gfx950", reason="gfx950 MXFP4 V validation")
@parametrize("q_format", [AttentionFormat.MXFP4, AttentionFormat.MXFP6])
def test_mha_v4_raw_mxfp4_v_supports_unaligned_sequence(q_format):
    torch.manual_seed(37)
    q = torch.randn((1, 129, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 257, 2, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    eager = mha_v4(q, k, v, q_format, q_format, AttentionFormat.MXFP4)
    compiled = torch.compile(mha_v4, fullgraph=True)(
        q, k, v, q_format, q_format, AttentionFormat.MXFP4
    )
    torch.cuda.synchronize()

    assert torch.equal(eager, compiled)
    assert torch.isfinite(compiled).all()
    # Determinism and finiteness alone pass on a wrong-but-stable result, so pin the value too.
    assert _cosine_against_attention(eager, q, k, v) > _MX_UNALIGNED_COSINE


def _cosine_against_attention(actual, q, k, v):
    """Cosine of a BSHD MHA v4 result against full-precision attention."""
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float(),
        v.transpose(1, 2).float(),
    ).transpose(1, 2)
    return torch.nn.functional.cosine_similarity(
        actual.float().flatten(), reference.flatten(), dim=0
    ).item()


# Loose enough to absorb MXFP4 quantization error, which costs about 0.02 on its own at a
# tile-aligned length, and tight enough that a mishandled partial tile cannot hide.
_MX_UNALIGNED_COSINE = 0.95

_MX_V_RECIPES = [
    param(native_fp8_format(), AttentionFormat.MXFP6, id="f8f6"),
    param(AttentionFormat.MXFP6, AttentionFormat.MXFP6, id="mxfp6"),
    param(AttentionFormat.MXFP6, AttentionFormat.MXFP4, id="f6f4"),
    param(AttentionFormat.MXFP4, AttentionFormat.MXFP4, id="mxfp4"),
]


@skipif(get_gfx() != "gfx950", reason="gfx950 MX V validation")
@parametrize(("qk_format", "v_format"), _MX_V_RECIPES)
@parametrize("tail", [1, 8, 32, 64, 96, 127])
def test_mha_v4_mx_v_partial_kv_tile_matches_attention(qk_format, v_format, tail):
    """Every MX V recipe must stay correct when the last KV tile is partly filled.

    The KV length is one full 128-token tile plus `tail`, so the only thing varying is
    partial-tile occupancy. Recipes with a per-tensor FP8 V are flat across `tail`, so any
    dependence here belongs to the MX V path.
    """
    sequence = 128 + tail
    torch.manual_seed(1234)
    q = torch.randn((1, sequence, 5, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = mha_v4(q, k, v, qk_format, qk_format, v_format)
    torch.cuda.synchronize()

    assert torch.isfinite(actual).all()
    assert _cosine_against_attention(actual, q, k, v) > _MX_UNALIGNED_COSINE


def run_torch_mha_v4(q, k, v, softmax_scale):
    """Compute dense BSHD attention in FP32 for benchmark validation."""
    scores = (
        torch.matmul(
            q.transpose(1, 2).float(), k.transpose(1, 2).float().transpose(-1, -2)
        )
        * softmax_scale
    )
    return torch.matmul(
        torch.softmax(scores, dim=-1), v.transpose(1, 2).float()
    ).transpose(1, 2)


_TOL_ERR_RATIO = 0.05  # matches checkAllclose default tol_err_ratio


def _verdict(err):
    if err == 0:
        return "pass"
    return "warning" if err <= _TOL_ERR_RATIO else "failed"


@benchmark()
def benchmark_mha_v4(batch, sequence_q, sequence_k, heads, dtype):
    """Benchmark every dense MHA v4 recipe against one Torch reference.

    All recipes are timed side by side because they differ only in how Q/K/V are quantized, so
    the interesting number is what each costs at the same shape and what it costs in accuracy.
    The quantized rows time their quantizers too, since that is what mha_v4 runs per call.
    """
    head_dim = 128
    softmax_scale = head_dim**-0.5
    torch.manual_seed(batch + sequence_q + sequence_k + heads)
    q = torch.randn((batch, sequence_q, heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((batch, sequence_k, heads, head_dim), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    reference = run_torch_mha_v4(q, k, v, softmax_scale)

    fp8 = native_fp8_format()
    recipes = {
        "bf16": (AttentionFormat.BF16, AttentionFormat.BF16),
        "bf16fp8": (AttentionFormat.BF16, fp8),
        "i8fp8": (AttentionFormat.INT8, fp8),
        "fp8": (fp8, fp8),
        "f8f6": (fp8, AttentionFormat.MXFP6),
        "f6f8": (AttentionFormat.MXFP6_E2M3, fp8),
        "mxfp6": (AttentionFormat.MXFP6_E2M3, AttentionFormat.MXFP6),
        "f6f4": (AttentionFormat.MXFP6_E2M3, AttentionFormat.MXFP4),
        "mxfp4": (AttentionFormat.MXFP4, AttentionFormat.MXFP4),
    }
    candidates = {
        name: (
            lambda q_format=q_format, v_format=v_format: mha_v4(
                q,
                k,
                v,
                q_format,
                q_format,
                v_format,
                softmax_scale=softmax_scale,
            )
        )
        for name, (q_format, v_format) in recipes.items()
    }

    flops = 4 * batch * heads * sequence_q * sequence_k * head_dim
    elements = batch * heads * head_dim * (sequence_q * 2 + sequence_k * 2)
    nbytes = elements * q.element_size()
    ret = {"gfx": get_gfx()}
    for name, candidate in candidates.items():
        output, us = run_perftest(candidate)
        # Quantized recipes are expected to land well outside a bitwise bound; the err column is
        # what carries their accuracy, so compare loosely and let the number speak.
        # catastrophic_check turns NaN/Inf or delta > ref_max/2 into an AssertionError; catching it
        # here marks the row failed instead of aborting, so the table still shows every recipe.
        accuracy_error = None
        try:
            err = checkAllclose(
                reference,
                output.to(dtypes.fp32),
                rtol=1e-1,
                atol=1e-1,
                msg=f"{name}: dense",
                catastrophic_check=True,
            )
        except AssertionError as exc:
            err = float("nan")
            accuracy_error = str(exc)
            aiter.logger.error("%s: dense: %s", name, accuracy_error)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
        ret[f"{name} result"] = (
            "failed" if accuracy_error is not None else _verdict(err)
        )
    return ret


def main():
    if get_gfx() != "gfx950":
        aiter.logger.warning(
            "MHA v4 dense benchmark unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Benchmark the dense MHA v4 recipes",
    )
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[1])
    parser.add_argument("--sequence-q", type=int, nargs="*", default=[1024])
    parser.add_argument("--sequence-k", type=int, nargs="*", default=[1024, 4096])
    parser.add_argument("--heads", type=int, nargs="*", default=[8])
    parser.add_argument(
        "-d", "--dtype", type=dtypes.str2Dtype, nargs="*", default=[dtypes.bf16]
    )
    args = parser.parse_args()

    rows = []
    for batch, sequence_q, sequence_k, heads, dtype in itertools.product(
        args.batch, args.sequence_q, args.sequence_k, args.heads, args.dtype
    ):
        if dtype != dtypes.bf16:
            aiter.logger.warning("MHA v4 dense benchmark skips dtype %s", dtype)
            continue
        rows.append(benchmark_mha_v4(batch, sequence_q, sequence_k, heads, dtype))
    if rows:
        frame = pd.DataFrame(rows)
        aiter.logger.info(
            "MHA v4 dense summary (markdown):\n%s",
            frame.to_markdown(index=False),
        )

    # Raised after the table so every recipe is reported, not just the first bad one.
    failed = sorted(
        {
            key.rsplit(" ", 1)[0]
            for row in rows
            for key, value in row.items()
            if key.endswith(" result") and value == "failed"
        }
    )
    if failed:
        raise RuntimeError(
            f"MHA v4 dense correctness failed for {failed}; see the summary above"
        )


def _dense_reference(q, k, v, valid):
    """Attention for one batch over its first `valid` keys, in BSHD."""
    return torch.nn.functional.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k[:, :valid].permute(0, 2, 1, 3),
        v[:, :valid].permute(0, 2, 1, 3),
    ).permute(0, 2, 1, 3)


@skipif(get_gfx() != "gfx950", reason="gfx950 per-batch key lengths")
@parametrize("lengths", [(300, 137), (512, 64), (1, 511), (65, 65)])
def test_mha_v4_seqlens_k_attends_over_each_batch_length(lengths):
    torch.manual_seed(41)
    batch, sequence, heads = len(lengths), 512, 4
    q = torch.randn((batch, sequence, heads, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    seqlens_k = torch.tensor(lengths, device="cuda", dtype=torch.int32)

    out = mha_v4(
        q,
        k,
        v,
        AttentionFormat.BF16,
        AttentionFormat.BF16,
        AttentionFormat.BF16,
        seqlens_k=seqlens_k,
    )
    torch.cuda.synchronize()

    for b, valid in enumerate(lengths):
        reference = _dense_reference(q[b : b + 1], k[b : b + 1], v[b : b + 1], valid)
        cosine = torch.nn.functional.cosine_similarity(
            out[b : b + 1].float().flatten(), reference.float().flatten(), dim=0
        )
        assert cosine > 0.99, f"batch {b} of {lengths}"


@skipif(get_gfx() != "gfx950", reason="gfx950 per-batch key lengths")
def test_mha_v4_seqlens_k_at_full_length_is_the_dense_result():
    """A null slot and a slot naming the whole key length must run the same code path."""
    torch.manual_seed(41)
    q = torch.randn((2, 512, 4, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    formats = (AttentionFormat.BF16, AttentionFormat.BF16, AttentionFormat.BF16)

    dense = mha_v4(q, k, v, *formats)
    full = mha_v4(
        q,
        k,
        v,
        *formats,
        seqlens_k=torch.full((2,), 512, device="cuda", dtype=torch.int32),
    )
    torch.cuda.synchronize()
    assert torch.equal(dense, full)


@skipif(get_gfx() != "gfx950", reason="dense varlen is a gfx950 row")
def test_mha_v4_rejects_unusable_seqlens_k():
    """Off gfx950 the capability gate preempts these, so the dtype and shape messages
    are only reachable here. The sparse rejection below is covered arch-independently
    by test_mha_v4_sparse_rejects_per_batch_key_lengths."""
    q = torch.randn((2, 256, 4, 128), device="cuda", dtype=torch.bfloat16)
    formats = (AttentionFormat.BF16, AttentionFormat.BF16, AttentionFormat.BF16)

    with expect_raises(ValueError, match="int32"):
        mha_v4(
            q, q, q, *formats, seqlens_k=torch.ones(2, device="cuda", dtype=torch.int64)
        )
    with expect_raises(ValueError, match="one entry per batch"):
        mha_v4(
            q, q, q, *formats, seqlens_k=torch.ones(1, device="cuda", dtype=torch.int32)
        )
    with expect_raises(NotImplementedError, match="per-batch key lengths"):
        mha_v4(
            q,
            q,
            q,
            AttentionFormat.INT8,
            AttentionFormat.INT8,
            native_fp8_format(),
            block_mask=torch.ones(
                (2, 4, 1, 256 // mha_v4_kv_tile()), device="cuda", dtype=torch.bool
            ),
            seqlens_k=torch.ones(2, device="cuda", dtype=torch.int32),
        )


def _launch_bf16_dense(q, out, seqlens_k=None, lse=None):
    """Drive the dense launcher directly, past the screening mha_v4_packed does first."""
    bf16 = int(AttentionFormat.BF16)
    torch.ops.aiter.mha_v4_fwd_launch(
        q,
        q,
        q,
        q,
        q,
        q,
        out,
        bf16,
        bf16,
        bf16,
        int(AttentionPack.DEFAULT),
        0,
        0,
        0,
        128**-0.5,
        seqlens_k,
        lse,
    )


@skipif(get_gfx() != "gfx950", reason="gfx950 per-batch key lengths")
def test_mha_v4_launch_rejects_off_device_seqlens_k():
    """The launcher hands this pointer straight to the GPU, so it guards independently of Python."""
    q = torch.randn((2, 256, 4, 128), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)

    with expect_raises(RuntimeError, match="same device as Q"):
        _launch_bf16_dense(q, out, seqlens_k=torch.full((2,), 256, dtype=torch.int32))


@skipif(get_gfx() != "gfx950", reason="gfx950 LSE")
def test_mha_v4_launch_rejects_off_device_lse():
    """mha_v4_packed takes an LSE buffer from the caller and never checks where it lives."""
    q = torch.randn((2, 256, 4, 128), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)

    with expect_raises(RuntimeError, match="same device as Q"):
        _launch_bf16_dense(q, out, lse=torch.empty((2, 4, 256), dtype=torch.float32))


def check_varlen_capable_objects(arch="gfx950"):
    """Every dense row _VARLEN_CAPABLE_QV claims must really read the per-batch length.

    The gate is keyed on (q_format, v_format) alone, so a pair that gains a second object -- a
    different head dim, say -- would inherit the claim without being rebuilt. Nothing numeric
    catches that: the kernel just attends over the full padded key length and returns the dense
    answer. Only the emitted object can refute it.
    """
    asm_dir = os.environ.get(
        "AITER_ASM_DIR", os.path.join(mha_v4_module.AITER_ROOT_DIR, "hsa")
    )
    root = os.path.join(asm_dir, arch, "fmha_v4_fwd")
    with open(os.path.join(root, "fmha_v4_fwd.csv"), encoding="utf-8") as manifest:
        rows = [
            line.strip().split(",")
            for line in manifest
            if line.strip() and not line.startswith("#")
        ]
    header, rows = rows[0], rows[1:]
    q_at, v_at = header.index("q_format"), header.index("v_format")
    mode_at, co_at = header.index("mode"), header.index("co_name")

    claimed = sorted(
        {
            row[co_at]
            for row in rows
            if row[mode_at] == "0"
            and (AttentionFormat(int(row[q_at])), AttentionFormat(int(row[v_at])))
            in mha_v4_module._VARLEN_CAPABLE_QV
        }
    )
    assert claimed, f"no dense {arch} rows are claimed varlen-capable"

    silent = []
    for name in claimed:
        listing = subprocess.run(
            [_llvm_objdump(), "-d", f"--mcpu={arch}", os.path.join(root, name)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        # Drop the trailing "// address: encoding" first, or a code address matches 0x1c0.
        if not re.search(r"\b0x1c0\b", re.sub(r"//.*", "", listing)):
            silent.append(name)
    assert not silent, (
        f"{arch} rows claimed in _VARLEN_CAPABLE_QV never load kernarg 0x1C0: {silent}; "
        "they would ignore seqlens_k and silently attend over the full padded key length"
    )
    aiter.logger.info(
        "varlen gate: all %d claimed %s rows load kernarg 0x1C0", len(claimed), arch
    )


@skipif(get_gfx() != "gfx950", reason="gfx950 MHA v4 validation")
@parametrize("arch", ["gfx950"])
def test_mha_v4_varlen_capable_rows_load_the_seqlens_k_kernarg(arch):
    check_varlen_capable_objects(arch)


if __name__ == "__main__":
    main()
    # CI runs this file as a script, so the checks only execute if main() invokes them.
    if _CHECKS.run(globals()):
        raise SystemExit(1)
