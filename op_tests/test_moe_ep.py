# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import (
    checkAllclose,
    run_perftest,
    perftest,
)
from aiter.fused_moe import (
    fused_topk,
    fused_moe,
    torch_moe,
)

from aiter.fused_moe_bf16_asm import asm_moe
from aiter.ops.shuffle import shuffle_weight
from aiter import ActivationType
from aiter import pertoken_quant
from aiter import dtypes
import argparse

BLOCK_SIZE_M = 32
MAX_TOKENS = 4096 * 4


@perftest(num_warmup=1, num_iters=2)
def torch_moe_test(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert, inter_dim, 1]
    fc2_scale=None,  # [expert, model_dim, 1]
    fc1_smooth_scale=None,  # [expert, 1, model_dim]
    fc2_smooth_scale=None,  # [expert, 1, inter_dim]
    expert_mask=None,
):
    return torch_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        fc1_scale,
        fc2_scale,
        fc1_smooth_scale,
        fc2_smooth_scale,
        expert_mask,
    )


@perftest()
def asm_moe_test(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert, inter_dim, 1]
    fc2_scale=None,  # [expert, model_dim, 1]
    fc1_smooth_scale=None,  # [expert, 1, model_dim]
    fc2_smooth_scale=None,  # [expert, 1, inter_dim]
    a16=False,
    expert_mask=None,
):

    return asm_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        fc1_scale,
        fc2_scale,
        fc1_smooth_scale,
        fc2_smooth_scale,
        a16,
        expert_mask=expert_mask,
    )


quant_algo = [
    "No",  # g1u0/ck(g1ux) support
    "int8quant",  # g1u1 support
    "fp8quant",  # g1u1 support
    "int8smoothquant",  # g1u1/g1u0 support
    "lqq",  # g1u1 support
    "fp8smoothquant",  # g1u1 support
]


def test_fmoe_ep(
    dtype,
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    quant="No",
    use_g1u1=False,
    shared_E=2,
    ep=8,
):
    # This gpu id in EP, this example use the last id
    ep_id = ep - 1
    # total_expert = unshared_expert + shared_expert + fake_expert(only use this fake expert id to mask)
    # expert_mask = torch.randint(
    #     0, 2, (E + shared_E + 1,), dtype=dtypes.i32, device="cuda"
    # )
    expert_mask = torch.zeros((E + shared_E + 1,), dtype=dtypes.i32, device="cuda")
    expert_mask[ep_id * (E // ep) : (ep_id + 1) * E // ep] = 1
    # # Get local expert Number in this gpu
    local_E = torch.sum(expert_mask).item()
    # The last expert
    fake_expertid = expert_mask.numel() - 1
    # Ensure fake expert to be masked
    expert_mask[-1] = 0
    # Ensure shared expert not to be masked
    expert_mask[E:-1] = 1

    quantAlgoId = quant_algo.index(quant)
    if quantAlgoId not in [0, 3] and not use_g1u1:
        print("g1u0 only could test no quant and int8smoothquant")
        return

    quantstr = quant_algo[quantAlgoId]
    quant_dtype = dtypes.i8 if quantstr.startswith("int8") else dtypes.fp8
    use_smooth = "smooth" in quantstr

    input = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    if use_g1u1:
        w1 = (
            torch.randn(
                (local_E + shared_E, inter_dim * 2, model_dim),
                dtype=dtype,
                device="cuda",
            )
            / 10
        )
    else:
        w1 = (
            torch.randn(
                (local_E + shared_E, inter_dim, model_dim), dtype=dtype, device="cuda"
            )
            / 10
        )
    w2 = (
        torch.randn(
            (local_E + shared_E, model_dim, inter_dim), dtype=dtype, device="cuda"
        )
        / 10
    )
    score = torch.randn((token, E), device="cuda", dtype=dtype)

    if shared_E > 0:
        shared_E_score = 0.1
        # init total_topk_ids, inference time you just need to fill ns_topk_ids in total_topk_ids
        total_topk_ids = torch.empty(
            (MAX_TOKENS, topk + shared_E + 1), dtype=dtypes.i32, device=input.device
        )
        ns_topk_ids, s_topk_ids = total_topk_ids.split([topk, shared_E + 1], dim=1)
        shared_expert_ids = [E + i for i in range(shared_E + 1)]
        s_topk_ids_list = [[fake_expertid] * (shared_E + 1)] * MAX_TOKENS
        for i in range(ep_id, MAX_TOKENS, ep):
            s_topk_ids_list[i] = shared_expert_ids
        s_topk_ids[:] = torch.tensor(
            s_topk_ids_list, dtype=dtypes.i32, device=input.device
        )

        # init total_topk_weights, inference time you just need to fill ns_topk_weights in total_topk_weights
        total_topk_weights = torch.empty(
            (MAX_TOKENS, topk + shared_E + 1), dtype=dtypes.fp32, device=input.device
        )
        ns_topk_weights, s_topk_weights = total_topk_weights.split(
            [topk, shared_E + 1], dim=1
        )
        s_topk_weights[:] = shared_E_score

        # inference time, use fused_topk to fill ns_topk_ids and ns_topk_weights
        fused_topk(input, score, topk, True, ns_topk_ids, ns_topk_weights)
        # inference time, topk_ids simply slices total_topk_ids into the number of input tokens, same for topk_weights
        topk_ids = total_topk_ids[:token]
        topk_weights = total_topk_weights[:token]

    else:
        topk_weights, topk_ids = fused_topk(input, score, topk, True)

    if quantAlgoId == 0:
        # ref2 implement
        ref2, avg_c = torch_moe_test(
            input, w1, w2, topk_weights, topk_ids, expert_mask=expert_mask
        )

        # b implement
        torch_quant = aiter.get_torch_quant(aiter.QuantType.No)
        w1_qt, w1_scale = torch_quant(w1, quant_dtype=None)
        w2_qt, w2_scale = torch_quant(w2, quant_dtype=None)
        w1_qt = w1_qt_aiter = w1_qt.view(w1.shape)
        w2_qt = w2_qt_aiter = w2_qt.view(w2.shape)
        w1_qt_aiter = shuffle_weight(w1_qt_aiter, layout=(16, 16))
        w2_qt_aiter = shuffle_weight(w2_qt_aiter, layout=(16, 16))

        # if use_g1u1:
        #     out_b = ref2
        #     avg_b = 9999
        #     print("asm g1u1 only support quant/smoothquant Now")
        # else:
        #     out_b, avg_b = asm_moe_test(
        #         input,
        #         w1_qt_aiter,
        #         w2_qt_aiter,
        #         topk_weights,
        #         topk_ids,
        #         expert_mask=expert_mask,
        #     )

        # test ck moe
        out_ck, avg_ck = run_perftest(
            fused_moe,
            input,
            w1_qt_aiter,
            w2_qt_aiter,
            topk_weights,
            topk_ids,
            expert_mask,
            w1_scale=None,
            w2_scale=None,
            quant_type=aiter.QuantType.No,
            activation=ActivationType.Silu,
            doweight_stage1=False,
        )

        # msg = f"[perf] {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, {ep=}, dtype: {dtype}, torch_avg: {avg_c:<8.2f} us, asm_avg: {avg_b:>8.2f} us, ck_avg: {avg_ck:>8.2f} us, uplift: {avg_c/avg_b-1:.1%}"
        # checkAllclose(ref2, out_b, rtol=0.01, atol=10, msg=msg)
        checkAllclose(ref2, out_ck, rtol=0.01, atol=10, msg="ck check")

    else:
        w1, fc1_scale = pertoken_quant(w1, quant_dtype=quant_dtype)
        w2, fc2_scale = pertoken_quant(w2, quant_dtype=quant_dtype)

        sp1 = (local_E + shared_E, inter_dim)
        sp2 = (local_E + shared_E, model_dim)

        if not use_smooth:
            fc1_smooth_scale = None
            fc2_smooth_scale = None
        else:
            # [expert, 1, model_dim]
            fc1_smooth_scale = torch.randn(sp2, dtype=dtypes.fp32, device="cuda")
            # [expert, 1, inter_dim]
            fc2_smooth_scale = torch.randn(sp1, dtype=dtypes.fp32, device="cuda")

        # ref2 implement
        ref2, avg_c = torch_moe_test(
            input,
            w1,
            w2,
            topk_weights,
            topk_ids,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
            expert_mask,
        )

        # b implement
        w1b = shuffle_weight(w1)
        w2b = shuffle_weight(w2)
        out_b, avg_b = asm_moe_test(
            input,
            w1b,
            w2b,
            topk_weights,
            topk_ids,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
            expert_mask=expert_mask,
        )

        def calculateTensorsSize(*args):
            num_btype = 0
            for el in args:
                if isinstance(el, torch.Tensor):
                    num_btype += el.element_size() * el.numel()
            return num_btype

        num_tb = calculateTensorsSize(
            input,
            input,
            w1b,
            w2b,
            topk_weights,
            topk_ids,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
        ) / (1024 * 1024 * 1024 * 1024.0)
        bw = num_tb * 1e6 / avg_b
        print(
            f"[BW  ] {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, {ep=}, {topk=}, dtype: {dtype}, asm_bandwidth: {bw:>8.2f}TB/s"
        )

        if use_smooth and (
            (
                (inter_dim % 512 == 0 or inter_dim % 320 == 0)
                and (w1b.dtype == dtypes.fp8 and inter_dim * 2 == w1b.shape[1])
            )
            or (
                (inter_dim % 256 == 0 or inter_dim % 320 == 0 or inter_dim % 384 == 0)
                and (w1b.dtype == dtypes.i8 and inter_dim * 2 == w1b.shape[1])
            )
            or (
                (inter_dim % 512 == 0)
                and (w1b.dtype == dtypes.i8 and inter_dim == w1b.shape[1])
            )
        ):
            out_b2, avg_b2 = asm_moe_test(
                input,
                w1b,
                w2b,
                topk_weights,
                topk_ids,
                fc1_scale,
                fc2_scale,
                fc1_smooth_scale,
                fc2_smooth_scale,
                a16=True,
                expert_mask=expert_mask,
            )
            msg = f"[perf] a8w8 asm: {avg_b:>8.2f} vs a16w8 asm: {avg_b2:>8.2f} ......"
            checkAllclose(out_b, out_b2, atol=10, msg=msg)

        msg = f"[perf] {use_g1u1=} {token=}, quant={quantstr}, {model_dim=}, {inter_dim=}, {E=}, {shared_E=}, {topk=}, {ep=}, {topk=}, dtype: {dtype}, torch_avg: {avg_c:<8.2f} us, asm_avg: {avg_b:>8.2f} us ...... uplift: {avg_c/avg_b-1:.1%}"
        checkAllclose(ref2, out_b, rtol=0.01, atol=10, msg=msg)
        # checkAllclose(ref2, avg_ck, rtol=0.01, atol=10)


import os
import struct
import numpy as np
from ctypes import c_int8
from aiter.int4_utils import convert_int8_to_uint32_int4, rearrange_4bit_elements
from aiter.utility import fp4_utils
from typing import Optional, Tuple
from aiter.fused_moe import (
    fused_topk,
    fused_moe,
    torch_moe_stage1,
    torch_moe_stage2,
)


def FloatMapToInt(tensor: torch.Tensor) -> torch.Tensor:
    byte_data = tensor.cpu().numpy().tobytes()
    int_array = np.frombuffer(byte_data, dtype=np.int32)
    arr_rshpd = int_array.reshape(tensor.shape)
    arr_writable = arr_rshpd.copy()
    int_tensor = torch.from_numpy(arr_writable).to(torch.int32)

    return int_tensor


def moe_init_lqq(
    eprt: int = 1,
    M: int = 128,
    N: int = 128,
    init_pattern: int = 0,
    min_val: int = -128,
    max_val: int = 127,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:

    qscale_buffer = torch.zeros((eprt, M, N), dtype=torch.int8)
    qzero_buffer = torch.zeros((eprt, M, N), dtype=torch.int8)

    qzero_range = max_val - min_val + 1

    offsets = torch.arange(eprt * M * N).reshape(eprt, M, N)

    temp_var0 = torch.zeros((eprt, M, N))
    temp_var1 = torch.zeros((eprt, M, N))
    d_idx = torch.arange(N).reshape(1, 1, N).expand(eprt, M, N)
    s_idx = torch.arange(M).reshape(1, M, 1).expand(eprt, M, N)

    if init_pattern == 0:
        temp_var0 = torch.randn((eprt, M, N))
        temp_var1 = torch.randn((eprt, M, N))

    elif init_pattern == 1:
        temp_var0 = torch.cos(offsets.float())
        temp_var1 = torch.cos(offsets.float())
        # temp_var0 = torch.full((eprt, M, N), 0.0, dtype=torch.float32)
        # temp_var1 = torch.full((eprt, M, N), 0.0, dtype=torch.float32)

    elif init_pattern == 2:
        temp_var0 = torch.sin(offsets.float())
        temp_var1 = temp_var0.clone()

    elif init_pattern == 3:
        temp_var0 = torch.cos(offsets.float()) + torch.sin(offsets.float())
        temp_var1 = temp_var0.clone()

    elif init_pattern == 10:
        temp_var0 = torch.full((eprt, M, N), 0.25)
        temp_var1 = temp_var0.clone()

    elif init_pattern == 11:
        temp_var0 = 0.01 * d_idx.float()
        temp_var1 = temp_var0.clone()

    elif init_pattern == 12:
        temp_var0 = 0.01 * s_idx.float()
        temp_var1 = temp_var0.clone()

    else:
        temp_var0 = torch.zeros((eprt, M, N))
        temp_var1 = temp_var0.clone()

    temp_var0_int = FloatMapToInt(temp_var0.cpu())
    temp_var1_int = FloatMapToInt(temp_var1.cpu())

    qzero_val_int = (temp_var0_int & 0xFF) % qzero_range + min_val
    qzero_buffer.copy_(qzero_val_int.to(torch.int8))

    maxI8_lo = qzero_val_int + 16
    maxI8_lo = torch.clamp(maxI8_lo, min_val, max_val)

    range_maxI8 = 119 - maxI8_lo + 1
    range_maxI8 = torch.clamp(range_maxI8, min=1)

    maxI8_val = (temp_var1_int & 0xFF) % range_maxI8 + maxI8_lo
    qscale_val_float = (maxI8_val - qzero_val_int).float() / 15.0
    qscale_val_rounded = torch.round(qscale_val_float)

    qscale_val_clamped = torch.clamp(qscale_val_rounded, min_val, max_val)
    qscale_buffer.copy_(qscale_val_clamped.to(torch.int8))

    return qscale_buffer.to(device), qzero_buffer.to(device)


def moe_init_uint4(
    eprt: int = 1,
    M: int = 128,
    N: int = 128,
    init_pattern: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:

    buffer = torch.zeros((eprt, M, N), dtype=torch.int8)

    offsets = torch.arange(eprt * M * N).reshape(eprt, M, N)

    d_idx = torch.arange(N).reshape(1, 1, N).expand(eprt, M, N).float()
    s_idx = torch.arange(M).reshape(1, M, 1).expand(eprt, M, N).float()

    temp_var = torch.zeros((eprt, M, N), dtype=torch.float32)

    if init_pattern == 0:
        temp_var = torch.randn((eprt, M, N), dtype=torch.float32)

    elif init_pattern == 1:
        temp_var = torch.cos(offsets.float())

    elif init_pattern == 2:
        temp_var = torch.sin(offsets.float())

    elif init_pattern == 3:
        temp_var = torch.cos(offsets.float()) + torch.sin(offsets.float())

    elif init_pattern == 10:
        temp_var = torch.full((eprt, M, N), 0.25, dtype=torch.float32)

    elif init_pattern == 11:
        temp_var = 0.01 * d_idx

    elif init_pattern == 12:
        temp_var = 0.01 * s_idx

    else:
        pass

    value32 = FloatMapToInt(temp_var.cpu())
    uint4_value = value32 & 0x0F
    buffer.copy_(uint4_value.to(torch.int8))

    return buffer.to(device)


def moe_init_int8(
    eprt: int = 1,
    M: int = 128,
    N: int = 128,
    init_pattern: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:

    buffer = torch.zeros((eprt, M, N), dtype=torch.int8)

    offsets = torch.arange(eprt * M * N).reshape(eprt, M, N)

    d_idx = torch.arange(N).reshape(1, 1, N).expand(eprt, M, N).float()
    s_idx = torch.arange(M).reshape(1, M, 1).expand(eprt, M, N).float()

    temp_var = torch.zeros((eprt, M, N), dtype=torch.float32)

    if init_pattern == 0:
        temp_var = torch.randn((eprt, M, N), dtype=torch.float32)

    elif init_pattern == 1:
        temp_var = torch.cos(offsets.float())

    elif init_pattern == 2:
        temp_var = torch.sin(offsets.float())

    elif init_pattern == 3:
        temp_var = torch.cos(offsets.float()) + torch.sin(offsets.float())

    elif init_pattern == 10:
        temp_var = torch.full((eprt, M, N), 0.25, dtype=torch.float32)

    elif init_pattern == 11:
        temp_var = 0.01 * d_idx

    elif init_pattern == 12:
        temp_var = 0.01 * s_idx

    else:
        pass

    value32 = FloatMapToInt(temp_var.cpu())
    uint8_value = value32 & 0xFF
    buffer.copy_(uint8_value.to(torch.int8))

    return buffer.to(device)


def moe_init_float(
    eprt,
    M,
    N,
    init_pattern=0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    total_size = eprt * M * N

    buffer = torch.zeros((eprt, M, N), dtype=torch.float32)
    indices = np.arange(total_size)

    b_indices = indices // (eprt * M * N)
    remainder = indices % (eprt * M * N)
    h_indices = remainder // (M * N)
    remainder = remainder % (M * N)
    s_indices = remainder // N
    d_indices = remainder % N

    if init_pattern == 0:
        values = np.random.randn(total_size).astype(np.float32)
    elif init_pattern == 1:
        values = np.cos(indices).astype(np.float32)
    elif init_pattern == 2:
        values = np.sin(indices).astype(np.float32)
    elif init_pattern == 3:
        values = np.cos(indices).astype(np.float32) + np.sin(indices).astype(np.float32)
    elif init_pattern == 10:
        values = np.full(total_size, 0.25, dtype=np.float32)
    elif init_pattern == 11:
        values = 0.01 * d_indices.astype(np.float32)
    elif init_pattern == 12:
        values = 0.01 * s_indices.astype(np.float32)
    else:
        values = np.zeros(total_size, dtype=np.float32)

    buffer = torch.from_numpy(values)
    buffer = buffer.view(eprt, M, N).to(torch.float32)

    return buffer.to(device)


def moe_lqq_dequant(
    in_buffer: torch.Tensor,
    qscale_buf: torch.Tensor,
    qzero_buf: torch.Tensor,
    group_in_k_lqq: int = 64,
    output_dtype: torch.dtype = torch.int8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    eprt, M, N = in_buffer.shape
    numGroups = N // group_in_k_lqq

    assert qscale_buf.shape == (
        eprt,
        M,
        numGroups,
    ), f"qscale_buf shape mismatch: {qscale_buf.shape} != ({eprt}, {M}, {numGroups})"
    assert qzero_buf.shape == (
        eprt,
        M,
        numGroups,
    ), f"qzero_buf shape mismatch: {qzero_buf.shape} != ({eprt}, {M}, {numGroups})"
    assert (
        N % group_in_k_lqq == 0
    ), f"N={N} must be divisible by group_in_k_lqq={group_in_k_lqq}"

    scale_expanded = qscale_buf.unsqueeze(-1).expand(-1, -1, -1, group_in_k_lqq)
    zero_expanded = qzero_buf.unsqueeze(-1).expand(-1, -1, -1, group_in_k_lqq)

    in_reshaped = in_buffer.view(eprt, M, numGroups, group_in_k_lqq)

    out_reshaped = torch.addcmul(
        zero_expanded.to(torch.float32),
        in_reshaped.to(torch.float32),
        scale_expanded.to(torch.float32),
    )

    out = out_reshaped.view(eprt, M, N).to(output_dtype)

    return out.to(device)


def moe_lqq_dequant_xor(
    in_buffer: torch.Tensor,
    qscale_buf: torch.Tensor,
    qzero_buf: torch.Tensor,
    group_in_k_lqq: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    eprt, M, N = in_buffer.shape
    numGroups = N // group_in_k_lqq

    if in_buffer is None or qscale_buf is None or qzero_buf is None:
        print("Neglect moe_init_uint4 for NULL input")
        return None

    if N % group_in_k_lqq != 0:
        print("N should be divisible by group_in_k_lqq")

    numGroups = N // group_in_k_lqq

    in_buffer_reshaped = in_buffer.view(eprt, M, numGroups, group_in_k_lqq)

    qscale_expanded = qscale_buf.view(eprt, M, numGroups, 1)
    qzero_expanded = qzero_buf.view(eprt, M, numGroups, 1)

    qzero_int = qzero_expanded.to(torch.int32)

    in_int32 = in_buffer_reshaped.to(torch.int32)
    qscale_int32 = qscale_expanded.to(torch.int32)

    result_int32 = in_int32 * qscale_int32 + qzero_int

    result_int32 = torch.bitwise_xor(
        result_int32, torch.tensor(0x80, dtype=torch.int32)
    )

    result_int8 = result_int32.to(torch.int8)

    return result_int8.view(eprt, M, N)


def moe_shuffle_inter(
    buffer: torch.Tensor,
    eprt: int,
    M: int,
    N: int,
    interExp: int,
) -> torch.Tensor:

    if buffer is None or buffer.numel() == 0:
        print("Neglect moe_shuffle for NULL input")
        return buffer

    groupSize = (1 << interExp) * 2

    if N % groupSize != 0:
        raise ValueError(
            f"N({N}) is not divisible by {groupSize} for interleave shuffle"
        )

    numGroups = N // groupSize

    # Store original info
    original_shape = buffer.shape
    original_device = buffer.device
    original_dtype = buffer.dtype

    # Flatten and reshape to (eprt, M, N)
    if buffer.dim() != 3:
        buffer_flat = buffer.flatten()
        buffer_reshaped = buffer_flat.view(eprt, M, N)
    else:
        buffer_reshaped = buffer

    # Create shuffle pattern
    shuffle_pattern = torch.zeros(groupSize, dtype=torch.long, device=buffer.device)
    for k in range(groupSize):
        shuffle_pattern[k] = (k >> 1) + ((k & 1) << interExp)

    # Reshape to (eprt, M, numGroups, groupSize)
    reshaped = buffer_reshaped.view(eprt, M, numGroups, groupSize)

    # Apply shuffle using gather
    shuffled = torch.gather(
        reshaped, dim=-1, index=shuffle_pattern.expand_as(reshaped).to(torch.long)
    )

    # Reshape back
    result = shuffled.view(eprt, M, N)

    # Return to original shape
    if buffer.dim() != 3:
        result = result.view(original_shape)

    return result


def moe_shuffle_32_16(x, eprt, M, N):
    tile_size_major = 32
    tile_size_minor = 16
    tile_stride = tile_size_major * tile_size_minor

    assert (
        M % tile_size_minor == 0
    ), f"M({M}) must be divisible by tileSizeMinor({tile_size_minor})"
    assert (
        N % tile_size_major == 0
    ), f"N({N}) must be divisible by tileSizeMajor({tile_size_major})"

    tiles_in_M = M // tile_size_minor
    tiles_in_N = N // tile_size_major
    total_tiles = tiles_in_M * tiles_in_N

    x = x.view(eprt, tiles_in_M, tile_size_minor, tiles_in_N, tile_size_major)

    x = x.permute(0, 1, 3, 2, 4).contiguous()
    x = x.view(eprt, total_tiles, tile_size_minor, tile_size_major)

    x = x.view(eprt, M, N)
    return x


def moe_shuffle_4_16(x: torch.Tensor, layout=(4, 16), use_int4=False) -> torch.Tensor:
    x_type = x.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)

    BN = 16
    BK = 4
    K = 1

    assert (
        x.shape[-2] % BN == 0
    ), f"N dimension {x.shape[-2]} must be divisible by BN={BN}"
    assert (
        x.shape[-1] % BK == 0
    ), f"K dimension {x.shape[-1]} must be divisible by BK={BK}"

    x_ = x
    x_ = x_.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK)
    x_ = x_.permute(0, 1, 3, 2, 4)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)

    x_.is_shuffled = True
    return x_


def moe_pack_int4(
    in_buffer: torch.Tensor,
    eprt: int,
    M: int,
    N: int,
) -> Optional[torch.Tensor]:

    # Validate inputs
    if in_buffer is None:
        print("Neglect moe_init for NULL input/output")
        return None

    if in_buffer.dtype != torch.int8:
        print(f"Expected torch.int8 but got {in_buffer.dtype}")
        return None

    # Check shape consistency
    if tuple(in_buffer.shape) != (eprt, M, N):
        print(f"Shape mismatch: expected ({eprt}, {M}, {N}), got {in_buffer.shape}")
        return None

    # Check even number of elements in the last dimension
    if eprt * M * N % 2 != 0:
        print("The element number of buffer should be even.")
        return None

    # Calculate output shape
    total_elems = eprt * M * N
    out_len = total_elems // 2

    # Reshape input to 1D for easier processing
    flattened = in_buffer.view(-1)

    # Create output tensor
    out_buffer = torch.empty(out_len, dtype=torch.int8, device=in_buffer.device)

    # Vectorized packing
    # Extract lower 4 bits from even indices
    v0 = flattened[::2] & 0x0F
    # Extract lower 4 bits from odd indices
    v1 = flattened[1::2] & 0x0F

    # Shift v1 left by 4 bits and pack with v0
    out_buffer[:] = v0 | (v1 << 4)

    # Reshape to match expected output dimensions
    return out_buffer.view(eprt, M, N // 2)


def save_buffer_to_file(
    buffer: torch.Tensor,
    filename: str,
    format: str = "numpy",
    metadata: Optional[dict] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    base_name = os.path.splitext(filename)[0]

    if buffer.device.type != "cpu":
        buffer_cpu = buffer.cpu()
    else:
        buffer_cpu = buffer

    if format.lower() == "numpy":
        np.save(f"{base_name}.npy", buffer_cpu.numpy())

        if metadata:
            import json

            with open(f"{base_name}_meta.json", "w") as f:
                json.dump(metadata, f, indent=2)

    elif format.lower() == "torch":
        save_dict = {"buffer": buffer_cpu}
        if metadata:
            save_dict["metadata"] = metadata

        torch.save(save_dict, f"{base_name}.pt")

    elif format.lower() == "text":
        with open(f"{base_name}.txt", "w") as f:
            f.write(f"Shape: {buffer.shape}\n")
            f.write(f"DataType: {buffer.dtype}\n")
            f.write(f"Min: {buffer.min().item()}\n")
            f.write(f"Max: {buffer.max().item()}\n")

            if metadata:
                f.write("\nMetadata:\n")
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")

            f.write("\nData:\n")

            buffer_cpu_fp32 = buffer_cpu.to(torch.float32)
            flat_data = buffer_cpu_fp32.flatten().numpy()
            for i in range(len(flat_data)):
                f.write(f"{flat_data[i]:.4e} ")
                if (i + 1) % 16 == 0:
                    f.write("\n")

    elif format.lower() == "binary":
        with open(f"{base_name}.hex", "wb") as f:
            buffer_cpu.numpy().tofile(f)


def save_int8_to_file(
    buffer: torch.Tensor,
    filename: str,
    format: str = "numpy",
    metadata: Optional[dict] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    base_name = os.path.splitext(filename)[0]

    if buffer.device.type != "cpu":
        buffer_cpu = buffer.cpu()
    else:
        buffer_cpu = buffer

    if format.lower() == "numpy":
        np.save(f"{base_name}.npy", buffer_cpu.numpy())

        if metadata:
            import json

            with open(f"{base_name}_meta.json", "w") as f:
                json.dump(metadata, f, indent=2)

    elif format.lower() == "torch":
        save_dict = {"buffer": buffer_cpu}
        if metadata:
            save_dict["metadata"] = metadata

        torch.save(save_dict, f"{base_name}.pt")

    elif format.lower() == "text":
        with open(f"{base_name}.txt", "w") as f:
            f.write(f"Shape: {buffer.shape}\n")
            f.write(f"DataType: {buffer.dtype}\n")
            f.write(f"Min: {buffer.min().item()}\n")
            f.write(f"Max: {buffer.max().item()}\n")

            if metadata:
                f.write("\nMetadata:\n")
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")

            f.write("\nData:\n")

            buffer_cpu_fp32 = buffer_cpu.to(torch.int8)
            flat_data = buffer_cpu_fp32.flatten().numpy()
            for i in range(len(flat_data)):
                f.write(f"{flat_data[i]:d} ")
                if (i + 1) % 16 == 0:
                    f.write("\n")

    elif format.lower() == "binary":
        with open(f"{base_name}.hex", "wb") as f:
            buffer_cpu.numpy().tofile(f)


def load_binary_to_tensor(file_path, shape, dtype=torch.int8):
    with open(file_path, "rb") as f:
        data = f.read()

    np_array = np.frombuffer(
        data,
        dtype={
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.uint8: np.uint8,
        }.get(dtype, np.int8),
    )

    tensor = torch.from_numpy(np_array).to(dtype)

    return tensor.reshape(shape).to("cuda")


def test_fmoe_lqq(
    dtype,
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    quant="No",
    use_g1u1=False,
    shared_E=2,
    ep=8,
):
    # This gpu id in EP, this example use the last id
    ep_id = ep - 1
    # total_expert = unshared_expert + shared_expert + fake_expert(only use this fake expert id to mask)
    # expert_mask = torch.randint(
    #     0, 2, (E + shared_E + 1,), dtype=dtypes.i32, device="cuda"
    # )
    expert_mask = torch.zeros((E + shared_E + 1,), dtype=dtypes.i32, device="cuda")
    expert_mask[ep_id * (E // ep) : (ep_id + 1) * E // ep] = 1
    # # Get local expert Number in this gpu
    local_E = torch.sum(expert_mask).item()
    # The last expert
    fake_expertid = expert_mask.numel() - 1
    # Ensure fake expert to be masked
    expert_mask[-1] = 0
    # Ensure shared expert not to be masked
    expert_mask[E:-1] = 1

    quantAlgoId = quant_algo.index(quant)
    if quantAlgoId not in [0, 3] and not use_g1u1:
        print("g1u0 only could test no quant and int8smoothquant")
        return

    AQDType = dtypes.i8
    W2QDType = dtypes.i8
    quantstr = quant_algo[quantAlgoId]
    use_smooth = "smooth" in quantstr

    print("==========================================")
    print("[test] batch: ", token)
    print("[test] expr: ", local_E + shared_E, "topk: ", topk)
    print("[test] model_dim: ", model_dim, " inter_dim: ", inter_dim)

    input = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    score = torch.randn((token, E), device="cuda", dtype=dtype)

    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    if shared_E > 0:
        shared_E_score = 0.5
        s_topk_weights = torch.tensor(
            [
                [shared_E_score, shared_E_score],
            ]
            * token,
            dtype=dtypes.fp32,
            device=input.device,
        )
        topk_weights = torch.cat((topk_weights, s_topk_weights), dim=1)
        s_topk_ids = torch.tensor(
            [
                [E, E + 1],
            ]
            * token,
            dtype=dtypes.i32,
            device=input.device,
        )
        topk_ids = torch.cat((topk_ids, s_topk_ids), dim=1)

    # O: int8
    # X_buf:                  int8  -> a1_qt             -> dev_X
    # X_dqn_buf:              float -> a1_scale          -> dev_X_dqn_buf
    # GU_buf:                 int8  -> w1_qt
    # GU_dqn_buf:             float -> w1_scale          -> dev_GU_dqn_buf
    # GU_buf_uint4:           uint4 -> w1_lqq_uint4
    # GU_buf_pack:            uint4 -> w1_lqq_pack -> dev_GU
    # GU_qscale_lqq_buf:      int4  -> w1_lqq_scale
    #                         int4  -> w1_lqq_scale_shf  -> dev_Qscl
    # GU_qzero_lqq_buf:       int4  -> w1_lqq_zero
    # GU_qzero_lqq_buf_uint8: uint8 -> w1_lqq_zero_uint8
    #                         uint8 -> w1_lqq_zero_uint8_shf -> dev_Qzero
    group_in_k_lqq = 64
    eprt = local_E + shared_E
    GU_dqn_k_lqq = model_dim // group_in_k_lqq
    GU_dqn_n_lqq = inter_dim
    GU_dqn_lqq_size = eprt * GU_dqn_k_lqq * GU_dqn_n_lqq * 2
    sz_GU = eprt * model_dim * inter_dim * 2
    GU_dqn_k = 1
    GU_dqn_n = inter_dim
    GU_dqn_size = eprt * GU_dqn_k * GU_dqn_n * 2
    X_dqn_k = 1
    X_dqn_m = token

    # input quant for kernel
    a1_qt, a1_scale = aiter.pertoken_quant(input, quant_dtype=AQDType)
    """
    sm1_scale = a1_scale
    a1_scale = None
    a1_scale = torch.empty((token, topk, 1), device="cuda", dtype=dtypes.fp32)
    a1 = torch.empty(
        (token, topk, model_dim),
        dtype=dtypes.i8,
        device="cuda",
    )
    hidden_states = a1_qt.view(1, token, model_dim).expand(topk, -1, -1)
    # aiter.moe_smoothquant_fwd(
    #     a1, hidden_states, sm1_scale, topk_ids, a1_scale
    # )
    aiter.smooth_per_token_scaled_quant(
        a1, hidden_states, a1_scale, sm1_scale, topk_ids
    )
    a1 = a1.view(-1, model_dim)
    """
    a1_qt = moe_init_int8(1, token, model_dim, 1)
    a1_scale = moe_init_float(1, X_dqn_m, X_dqn_k, 1)
    a1_qt = a1_qt.reshape(token, model_dim)
    a1_scale = a1_scale.reshape(X_dqn_m, X_dqn_k)
    # save_buffer_to_file(a1_qt, "./feifei/a1_qt", format="binary")
    # save_buffer_to_file(a1_scale, "./feifei/a1_scale", format="text")

    # lqq init for kernel
    w1_lqq_scale, w1_lqq_zero = moe_init_lqq(
        eprt, GU_dqn_lqq_size // eprt // GU_dqn_k_lqq, GU_dqn_k_lqq, 1, -100, 100
    )
    # save_buffer_to_file(w1_lqq_scale, "./feifei/w1_lqq_scale", format="binary")
    # save_buffer_to_file(w1_lqq_zero, "./feifei/w1_lqq_zero", format="binary")
    w1_lqq_zero_uint8 = (w1_lqq_zero.to(torch.int16) + 128).to(torch.uint8)
    # save_buffer_to_file(
    #    w1_lqq_zero_uint8, "./feifei/w1_lqq_zero_uint8", format="binary"
    # )
    w1_lqq_uint4 = moe_init_uint4(eprt, sz_GU // model_dim // eprt, model_dim, 1)
    # save_buffer_to_file(w1_lqq_uint4, "./feifei/w1_lqq_uint4", format="binary")

    # shuffle w1 for kernel
    w1_lqq_uint4_shf_inter = moe_shuffle_inter(
        w1_lqq_uint4, eprt, sz_GU // model_dim // eprt, model_dim, 6
    )
    # save_buffer_to_file(
    #    w1_lqq_uint4_shf_inter, "./feifei/w1_lqq_uint4_shf_inter", format="binary"
    # )
    # save_int8_to_file(
    #    w1_lqq_uint4_shf_inter, "./feifei/w1_lqq_uint4_shf21", format="text"
    # )
    w1_lqq_uint4_shf2 = moe_shuffle_32_16(
        w1_lqq_uint4_shf_inter, eprt, sz_GU // model_dim // eprt, model_dim
    )
    # save_int8_to_file(w1_lqq_uint4_shf2, "./feifei/w1_lqq_uint4_shf22", format="text")
    # save_buffer_to_file(
    #    w1_lqq_uint4_shf2, "./feifei/w1_lqq_uint4_shf2", format="binary"
    # )
    w1_lqq_pack = moe_pack_int4(
        w1_lqq_uint4_shf2, eprt, sz_GU // model_dim // eprt, model_dim
    )
    w1_lqq = w1_lqq_pack.view(dtypes.i4x2)
    # save_buffer_to_file(w1_lqq_pack, "./feifei/w1_lqq_pack", format="binary")
    w1_lqq_scale_shf = moe_shuffle_4_16(w1_lqq_scale, (4, 16), use_int4=False)
    # save_buffer_to_file(w1_lqq_scale_shf, "./feifei/w1_lqq_scale_shf", format="binary")
    w1_lqq_zero_uint8_shf = moe_shuffle_4_16(w1_lqq_zero_uint8, (4, 16), use_int4=False)
    # save_buffer_to_file(
    #    w1_lqq_zero_uint8_shf, "./feifei/w1_lqq_zero_uint8_shf", format="binary"
    # )

    # gu quant for cpu ref
    w1_qt = moe_lqq_dequant(w1_lqq_uint4, w1_lqq_scale, w1_lqq_zero)
    # save_int8_to_file(w1_qt, "./feifei/w1_qt", format="text")
    # gu quant scale for cpu ref and kernel
    w1_scale = moe_init_float(eprt, GU_dqn_size // eprt // GU_dqn_k, GU_dqn_k, 1)
    # save_buffer_to_file(w1_scale, "./feifei/w1_scale", format="text")

    ######################################################################################
    # w2 = (
    #     torch.randn(
    #         (local_E + shared_E, model_dim, inter_dim),
    #         dtype=dtype,
    #         device="cuda",
    #     )
    #     / 10
    # )
    # w2_qt, w2_scale = aiter.pertoken_quant(w2, quant_dtype=dtypes.i8, dtypeMax=7)
    #######################################################################################
    D_dqn_n_lqq = model_dim
    D_dqn_k_lqq = inter_dim // group_in_k_lqq

    # lqq init for kernel
    w2_lqq_scale, w2_lqq_zero = moe_init_lqq(
        eprt, D_dqn_n_lqq, D_dqn_k_lqq, 1, -100, 100
    )

    # save_buffer_to_file(w2_lqq_scale, "./feifei/w2_lqq_scale", format="binary")
    # save_buffer_to_file(w2_lqq_zero, "./feifei/w2_lqq_zero", format="binary")
    w2_lqq_zero_uint8 = (w2_lqq_zero.to(torch.int16) + 128).to(torch.uint8)
    # save_buffer_to_file(
    #    w2_lqq_zero_uint8, "./feifei/w2_lqq_zero_uint8", format="binary"
    # )
    w2_lqq_uint4 = moe_init_uint4(eprt, model_dim, inter_dim, 1)
    # save_buffer_to_file(w2_lqq_uint4, "./feifei/w2_lqq_uint4", format="binary")

    # shuffle w2 for kernel
    w2_lqq_uint4_shf_inter = moe_shuffle_inter(
        w2_lqq_uint4, eprt, model_dim, inter_dim, 6
    )
    # save_buffer_to_file(
    #    w2_lqq_uint4_shf_inter, "./feifei/w2_lqq_uint4_shf_inter", format="binary"
    # )
    # save_int8_to_file(
    #    w2_lqq_uint4_shf_inter, "./feifei/w2_lqq_uint4_shf21", format="text"
    # )
    w2_lqq_uint4_shf2 = moe_shuffle_32_16(
        w2_lqq_uint4_shf_inter, eprt, model_dim, inter_dim
    )
    # save_int8_to_file(w2_lqq_uint4_shf2, "./feifei/w2_lqq_uint4_shf22", format="text")
    # save_buffer_to_file(
    #    w2_lqq_uint4_shf2, "./feifei/w2_lqq_uint4_shf2", format="binary"
    # )
    w2_lqq_pack = moe_pack_int4(w2_lqq_uint4_shf2, eprt, model_dim, inter_dim)
    w2_lqq = w2_lqq_pack.view(dtypes.i4x2)
    # save_buffer_to_file(w2_lqq_pack, "./feifei/w2_lqq_pack", format="binary")
    w2_lqq_scale_shf = moe_shuffle_4_16(w2_lqq_scale, (4, 16), use_int4=False)
    # save_buffer_to_file(w2_lqq_scale_shf, "./feifei/w2_lqq_scale_shf", format="binary")
    w2_lqq_zero_uint8_shf = moe_shuffle_4_16(w2_lqq_zero_uint8, (4, 16), use_int4=False)
    # save_buffer_to_file(
    #    w2_lqq_zero_uint8_shf, "./feifei/w2_lqq_zero_uint8_shf", format="binary"
    # )

    # gu quant for cpu ref
    w2_qt = moe_lqq_dequant_xor(w2_lqq_uint4, w2_lqq_scale, w2_lqq_zero_uint8)
    # save_int8_to_file(w2_qt, "./feifei/w2_qt", format="text")
    # gu quant scale for cpu ref and kernel
    w2_scale = moe_init_float(eprt, inter_dim, 1, 1)
    # save_buffer_to_file(w2_scale, "./feifei/w2_scale", format="text")

    print("==========================================")
    print("[test] a1_qt   : ", a1_qt.shape, a1_qt.dtype)
    print("[test] w1_qt   : ", w1_qt.shape, w1_qt.dtype)
    print("[test] w2_qt   : ", w2_qt.shape, w2_qt.dtype)
    print("[test] a1_scale: ", a1_scale.shape, a1_scale.dtype)
    print("[test] w1_scale: ", w1_scale.shape, w1_scale.dtype)
    print("[test] w2_scale: ", w2_scale.shape, w2_scale.dtype)

    out1_ref = torch_moe_stage1(
        a1_qt,
        w1_qt,
        w2_qt,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.per_Token,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        doweight=False,
    )
    print("[test] out1_ref: ", out1_ref.shape, out1_ref.dtype)

    print("------------------------------------------")
    a2_qt, a2_scale = aiter.pertoken_quant(out1_ref, quant_dtype=dtypes.i8, dtypeMax=7)
    a2_qt = a2_qt.view(token, topk, -1)
    print("[test] a2_qt   : ", a2_qt.shape, a2_qt.dtype)
    print("[test] a2_scale: ", a2_scale.shape, a2_scale.dtype)
    # out2_ref = torch_moe_stage2(
    #    a2_qt,
    #    w1_qt,
    #    w2_qt,
    #    topk_weights,
    #    topk_ids,
    #    dtype=dtype,
    #    quant_type=aiter.QuantType.per_Token,
    #    a2_scale=a2_scale,
    #    w2_scale=w2_scale,
    #    doweight=False,
    # )
    # print("[test] out2_ref: ", out2_ref.shape, out2_ref.dtype)
    # save_buffer_to_file(out1_ref, "./feifei/out1_ref", format="text")
    # save_int8_to_file(a2_qt, "./feifei/a2_qt", format="text")
    # save_buffer_to_file(a2_scale, "./feifei/a2_scale", format="text")

    print("==========================================")
    print("[test] a1_qt       : ", a1_qt.shape, a1_qt.dtype)
    print("[test] a1_scale    : ", a1_scale.shape, a1_scale.dtype)
    print("[test] w1_lqq      : ", w1_lqq.shape, w1_lqq.dtype)
    print("[test] w1_scale    : ", w1_scale.shape, w1_scale.dtype)
    print("[test] w1_lqq_scale: ", w1_lqq_scale.shape, w1_lqq_scale.dtype)
    print("[test] w1_lqq_zero : ", w1_lqq_zero.shape, w1_lqq_zero.dtype)
    print("[test] w2_qt       : ", w2_qt.shape, w2_qt.dtype)
    print("[test] w2_scale    : ", w2_scale.shape, w2_scale.dtype)
    out1_asm = fused_moe(
        a1_qt,
        w1_lqq,
        w2_lqq,
        topk_weights,
        topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        w1_lqq_scale=w1_lqq_scale_shf,
        w1_lqq_zero=w1_lqq_zero_uint8_shf,
        quant_type=aiter.QuantType.per_Token,
        activation=aiter.ActivationType.Silu,
        doweight_stage1=False,
        dtype=dtype,
        block_size_M=80,
    )
    print("[test] out1_asm    : ", out1_asm.shape, out1_asm.dtype)
    print("------------------------------------------")
    # save_buffer_to_file(out1_asm, "./feifei/out1_asm_fp32", format="text")
    # save_int8_to_file(out1_asm, "./feifei/out1_asm_int8", format="text")

    err = checkAllclose(out1_ref, out1_asm)
    print(err)

    """
    out1_asm, us1 = run_perftest(
        fused_moe,
        a1_qt,
        w1_lqq_pack,
        w2_qt,
        topk_weights,
        topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        w1_lqq_scale=w1_lqq_scale_shf,
        w1_lqq_zero=w1_lqq_zero_uint8_shf,
        quant_type=aiter.QuantType.per_Token,
        activation=aiter.ActivationType.Silu,
        doweight_stage1=False,
        dtype=dtype,
        block_size_M=80,
        num_iters=100,
        num_warmup=2,
    )
    err = checkAllclose(
        out1_ref,
        out1_asm,
        msg=f"asm_moe_stage1:{us1:>8.2f} us, {token*model_dim*inter_dim*3*topk*2/us1/1000/1000:>8.2f} tflops......(quant:{AQDType})",
    )
    """


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="select test",
)
l_test = [
    "test_fmoe_16_bit",
    "g1u1_no_quant",
    "g1u1_int8quant",
    "g1u1_fp8quant",
    "g1u0_int8smoothquant",
    "g1u1_int8smoothquant",
    "g1u1_fp8smoothquant",
    "g1u1_lqq",
]
parser.add_argument(
    "-t",
    "--test",
    type=str,
    choices=l_test,
    default=None,
    help="""Select test to run.
    e.g.: -t g1u1_int8quant
          or -t test_fmoe_16_bit
          or -t g1u1_no_quant
          or -t g1u1_int8quant
          or -t g1u1_fp8quant
          or -t g1u0_int8smoothquant
          or -t g1u1_int8smoothquant
          or -t g1u1_fp8smoothquant""",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    nargs="?",
    default=None,
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-m",
    "--token",
    type=int,
    nargs="*",
    default=None,
    help="""Token Num.
    e.g.: -m 128""",
)
parser.add_argument(
    "-hd",
    "--hidden_dim",
    type=int,
    nargs="*",
    default=None,
    help="""Hidden states dim.
    e.g.: -hd 4096""",
)
parser.add_argument(
    "-md",
    "--model_dim",
    type=int,
    nargs="*",
    default=None,
    help="""Model dim.
    e.g.: -md 4096""",
)
parser.add_argument(
    "-id",
    "--inter_dim",
    type=int,
    nargs="*",
    default=None,
    help="""Intermediate dim.
    e.g.: -id 1024""",
)
parser.add_argument(
    "-e",
    "--expert",
    type=int,
    nargs="?",
    default=None,
    help="""Number of experts.
    e.g.: -e 32""",
)
parser.add_argument(
    "-k",
    "--topk",
    type=int,
    nargs="?",
    default=None,
    help="""Top-k value.
    e.g.: -k 5""",
)
parser.add_argument(
    "-ep",
    "--expert_parallelism",
    type=int,
    nargs="*",
    default=None,
    help="""Expert Parallelism.
    e.g.: -ep 8""",
)
parser.add_argument(
    "-se",
    "--shared_expert",
    type=int,
    nargs="?",
    default=None,
    help="""Shared experts.
    e.g.: -se 0""",
)

args = parser.parse_args()
if args.test is not None:
    l_test = [args.test]

for test in l_test:
    print(f"\nRunning test: {test}")
    shared_E = 2 if args.shared_expert is None else args.shared_expert
    if test == "test_fmoe_16_bit":
        print("test test_fmoe 16 bit")
        # print("\ng1u0 no quant")
        # for dtype in [dtypes.fp16, dtypes.bf16]:
        #     for m in [7, 128, 256]:
        #         for dim in [4096, 8192]:
        #             for hdim in [1024, 1280]:
        #                 for ep in [4, 8]:
        #                     test_fmoe_ep(
        #                         dtype, m, dim, hdim, 128, 6, quant="No", shared_E=shared_E, ep=ep

    elif test == "g1u1_no_quant":
        for dtype in (
            [dtypes.fp16, dtypes.bf16]
            if args.dtype is None
            else [dtypes.d_dtypes[args.dtype]]
        ):
            for m in [7, 128, 256] if args.token is None else args.token:
                for hdim in (
                    [4096, 8192] if args.hidden_dim is None else args.hidden_dim
                ):
                    for idim in (
                        [1024, 1280] if args.inter_dim is None else args.inter_dim
                    ):
                        for ep in (
                            [4, 8]
                            if args.expert_parallelism is None
                            else args.expert_parallelism
                        ):
                            expert = 128 if args.expert is None else args.expert
                            topk = 9 if args.topk is None else args.topk
                            test_fmoe_ep(
                                dtype,
                                m,
                                hdim,
                                idim,
                                expert,
                                topk,
                                quant="No",
                                use_g1u1=True,
                                shared_E=shared_E,
                                ep=ep,
                            )
    elif test == "g1u1_int8quant":
        for dtype in (
            [dtypes.bf16] if args.dtype is None else [dtypes.d_dtypes[args.dtype]]
        ):
            for m in [128, 256] if args.token is None else args.token:
                for hdim in (
                    [4096, 8192] if args.hidden_dim is None else args.hidden_dim
                ):
                    for idim in [1024] if args.inter_dim is None else args.inter_dim:
                        expert = 32 if args.expert is None else args.expert
                        topk = 5 if args.topk is None else args.topk
                        for ep in (
                            [4, 8]
                            if args.expert_parallelism is None
                            else args.expert_parallelism
                        ):
                            test_fmoe_ep(
                                dtype,
                                m,
                                hdim,
                                idim,
                                expert,
                                topk,
                                quant="int8quant",
                                use_g1u1=True,
                                shared_E=shared_E,
                                ep=ep,
                            )
    elif test == "g1u1_fp8quant":
        for dtype in (
            [dtypes.bf16] if args.dtype is None else [dtypes.d_dtypes[args.dtype]]
        ):
            for m in [128, 256] if args.token is None else args.token:
                for hdim in (
                    [4096, 8192] if args.hidden_dim is None else args.hidden_dim
                ):
                    for idim in [1024] if args.inter_dim is None else args.inter_dim:
                        expert = 32 if args.expert is None else args.expert
                        topk = 5 if args.topk is None else args.topk
                        for ep in (
                            [4, 8]
                            if args.expert_parallelism is None
                            else args.expert_parallelism
                        ):
                            test_fmoe_ep(
                                dtype,
                                m,
                                hdim,
                                idim,
                                expert,
                                topk,
                                quant="fp8quant",
                                use_g1u1=True,
                                shared_E=shared_E,
                                ep=ep,
                            )
    elif test == "g1u0_int8smoothquant":
        for dtype in (
            [dtypes.bf16] if args.dtype is None else [dtypes.d_dtypes[args.dtype]]
        ):
            for m in [128] if args.token is None else args.token:
                for hdim in (
                    [4096, 6144, 8192] if args.hidden_dim is None else args.hidden_dim
                ):
                    for idim in (
                        [512, 1024] if args.inter_dim is None else args.inter_dim
                    ):
                        expert = 32 if args.expert is None else args.expert
                        topk = 5 if args.topk is None else args.topk
                        for ep in (
                            [4, 8]
                            if args.expert_parallelism is None
                            else args.expert_parallelism
                        ):
                            test_fmoe_ep(
                                dtype,
                                m,
                                hdim,
                                idim,
                                expert,
                                topk,
                                quant="int8smoothquant",
                                use_g1u1=False,
                                shared_E=shared_E,
                                ep=ep,
                            )
    elif test == "g1u1_int8smoothquant":
        for dtype in (
            [dtypes.bf16] if args.dtype is None else [dtypes.d_dtypes[args.dtype]]
        ):
            for m in [128] if args.token is None else args.token:
                for hdim in [4096] if args.hidden_dim is None else args.hidden_dim:
                    for idim in [1280] if args.inter_dim is None else args.inter_dim:
                        expert = 128 if args.expert is None else args.expert
                        topk = 6 if args.topk is None else args.topk
                        for ep in (
                            [8]
                            if args.expert_parallelism is None
                            else args.expert_parallelism
                        ):
                            test_fmoe_ep(
                                dtype,
                                m,
                                hdim,
                                idim,
                                expert,
                                topk,
                                quant="int8smoothquant",
                                use_g1u1=True,
                                shared_E=shared_E,
                                ep=ep,
                            )
    elif test == "g1u1_fp8smoothquant":
        for dtype in (
            [dtypes.bf16] if args.dtype is None else [dtypes.d_dtypes[args.dtype]]
        ):
            for m in [128] if args.token is None else args.token:
                for hdim in (
                    [4096, 6144, 8192] if args.hidden_dim is None else args.hidden_dim
                ):
                    for idim in (
                        [512, 1024, 1280] if args.inter_dim is None else args.inter_dim
                    ):
                        expert = 32 if args.expert is None else args.expert
                        topk = 5 if args.topk is None else args.topk
                        for ep in (
                            [4, 8]
                            if args.expert_parallelism is None
                            else args.expert_parallelism
                        ):
                            test_fmoe_ep(
                                dtype,
                                m,
                                hdim,
                                idim,
                                expert,
                                topk,
                                quant="fp8smoothquant",
                                use_g1u1=True,
                                shared_E=shared_E,
                                ep=ep,
                            )
    elif test == "g1u1_lqq":
        for dtype in (
            [dtypes.bf16] if args.dtype is None else [dtypes.d_dtypes[args.dtype]]
        ):
            for m in [128] if args.token is None else args.token:
                for mdim in [4096] if args.model_dim is None else args.model_dim:
                    for idim in [1280] if args.inter_dim is None else args.inter_dim:
                        expert = 128 if args.expert is None else args.expert
                        topk = 6 if args.topk is None else args.topk
                        for ep in (
                            [8]
                            if args.expert_parallelism is None
                            else args.expert_parallelism
                        ):
                            test_fmoe_lqq(
                                dtype,
                                m,
                                mdim,
                                idim,
                                expert,
                                topk,
                                quant="lqq",
                                use_g1u1=True,
                                shared_E=shared_E,
                                ep=ep,
                            )
    else:
        raise ValueError(f"Unknown test: {test}")
