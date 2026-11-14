# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for the gfx942 FlyDSL whole-MoE backend.

Its tuned rows use cu_num=80 (MI308X), so test_moe_2stage.py skips them on
runners with another CU count. This test runs every shipped config through the
registered backend on any gfx942 device, checks the compact Down task table
for both MI308X and MI300X CU counts, and checks the padded sorted reduction.
"""

import argparse
import copy
import functools
import itertools
import math
from unittest.mock import patch

import pandas as pd
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    AUX_SORT_OPUS,
    fused_moe,
    fused_topk,
    moe_sorting,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.fused_moe_registry import (
    FusedMoeRequest,
    make_fused_moe_impl_kernel_name,
    resolve_fused_moe_impl,
)
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.flydsl.fused_moe_gfx942 import (
    Config,
    _describe,
    _gateup_output,
    _kernel_block_scale,
    _Problem,
    block_scale_groups,
    flydsl_block_weight_scale,
    get_tune_space,
    run_flydsl_moe_gfx942,
)
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import (
    gemm2_8x1_compact as compact,
)
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import sorted_sum
from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.common import _ptr
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

IMPL = "flydsl_gfx942"
QUANTS = {
    "no": QuantType.No,
    "per_tensor": QuantType.per_Tensor,
    "ptpc": QuantType.per_Token,
    "block": QuantType.per_128x128,
}
ACTIVATIONS = {
    "silu": ActivationType.Silu,
    "swiglu": ActivationType.Swiglu,
    "swiglustep": ActivationType.SwigluStep,
}
# MI300X then MI308X, the order in which a cached launcher meets a smaller device.
CU_COUNTS = (304, 80)
GUARD, CANARY = 64, -123
# FP8 rounding moves an element by at most 0.28 token RMS from the torch reference.
ELEMENT_ERR_LIMIT = 1.0
# FP8 rounding reaches 0.076 per token; one misrouted route gives about 0.67.
TOKEN_ERR_LIMIT = 0.2
LOGITS_DIFF_LIMIT = 1e-3
# Prefill paths of one case share Gate/Up and quantization and agree to BF16 rounding.
PATH_DIFF_LIMIT = 0.1
SWIGLU_STEP_LIMIT = 7.0


def guarded_task_buffers(sorted_expert_ids, experts, cu_count):
    """allocate_task_buffers with canaries before and after every buffer."""
    full, tail = compact.task_capacities(sorted_expert_ids.numel(), experts, cu_count)
    storages, buffers = [], []
    for shape in ((full, 2), (tail, 2), (2,)):
        storage = torch.full(
            (torch.Size(shape).numel() + 2 * GUARD,),
            CANARY,
            dtype=torch.int32,
            device=sorted_expert_ids.device,
        )
        storages.append(storage)
        buffers.append(storage[GUARD:-GUARD].view(shape))
    return storages, buffers


def check_guards(storages):
    for storage in storages:
        assert bool((storage[:GUARD] == CANARY).all()), "write before a buffer"
        assert bool((storage[-GUARD:] == CANARY).all()), "write after a buffer"


def check_tasks(blocks, experts, cu_count, full_tasks, tail_tasks, counts):
    """Check the split policy and that each block is covered once by its expert."""
    sizes = [0] * experts
    for expert in blocks:
        sizes[expert] += 1
    full = sum(size // 4 for size in sizes)
    tail = sum(size % 4 for size in sizes)
    remainder = full % cu_count
    split = 0
    if remainder and remainder * compact._SPLIT_DEN < cu_count * compact._SPLIT_NUM:
        split = remainder
    # The kernel caps the split at the free tail capacity.
    split = min(split, (tail_tasks.shape[0] - tail) // 4)
    expected = [(full - split) * 256, (tail + 4 * split) * 64]
    assert counts.tolist() == expected, f"counts {counts.tolist()} != {expected}"
    coverage = [0] * len(blocks)
    for tasks, rows, count in (
        (full_tasks, 256, full - split),
        (tail_tasks, 64, tail + 4 * split),
    ):
        assert count <= tasks.shape[0], "task count exceeds its buffer"
        for start, expert in tasks[:count].tolist():
            assert start % 64 == 0 and 0 <= start and start + rows <= 64 * len(blocks)
            for block in range(start // 64, (start + rows) // 64):
                assert blocks[block] == expert, "task routed to another expert"
                coverage[block] += 1
    assert coverage == [1] * len(blocks), "blocks not covered exactly once"
    return full - split, tail + 4 * split


def task_scenarios(cu_count):
    """Map a scenario to (experts, [(expert, M64 blocks)]) in metadata order."""
    # Largest remainder of the final full-task CU round that is still split.
    split = (cu_count * compact._SPLIT_NUM - 1) // compact._SPLIT_DEN
    return {
        "empty": (4, []),
        "tails_only": (8, [(1, 1), (3, 2), (6, 3)]),
        "single_expert": (1, [(0, 8)]),
        "full_round": (2, [(1, 4 * cu_count)]),
        "split_limit": (2, [(0, 4 * (cu_count + split)), (1, 3)]),
        "no_split": (2, [(0, 4 * (cu_count + split + 1)), (1, 3)]),
        # Every expert has one full task and three tails, filling the tail capacity.
        "tail_capacity": (split, [(expert, 7) for expert in range(split)]),
        # 60 full tasks split into tails with 304 CUs but not with 80.
        "cu_dependent": (1, [(0, 240)]),
        "experts_2048": (2048, [(expert, 1 + expert % 7) for expert in range(2048)]),
    }


# One launcher per expert count serves every CU count, like the cached builder.
build_task_table = functools.cache(compact.build_task_table)


@benchmark()
def test_task_table(scenario, cu_count, buffer_cu_count):
    experts, runs = task_scenarios(cu_count)[scenario]
    blocks = [expert for expert, size in runs for _ in range(size)]
    # Padding names a real expert, so reading past the valid blocks breaks a run.
    metadata = torch.tensor(blocks + [0] * 4, dtype=torch.int32)
    valid_rows = torch.tensor([64 * len(blocks)], dtype=torch.int32)
    storages, (full_tasks, tail_tasks, counts) = guarded_task_buffers(
        metadata, experts, buffer_cu_count
    )
    launcher = build_task_table(experts)
    pointers = [_ptr(t) for t in (metadata, valid_rows, full_tasks, tail_tasks, counts)]

    def build():
        _run_compiled(
            launcher,
            *pointers,
            cu_count,
            tail_tasks.shape[0],
            torch.cuda.current_stream(),
        )
        return counts

    _, us = run_perftest(build)
    full, tail = check_tasks(blocks, experts, cu_count, full_tasks, tail_tasks, counts)
    check_guards(storages)
    if scenario == "tail_capacity" and buffer_cu_count == cu_count:
        assert tail == tail_tasks.shape[0], "tail capacity is not tight"
    nbytes = metadata.nbytes + valid_rows.nbytes + 8 * (full + tail) + counts.nbytes
    return {
        "blocks": len(blocks),
        "full tasks": full,
        "full capacity": full_tasks.shape[0],
        "tail tasks": tail,
        "tail capacity": tail_tasks.shape[0],
        "us": us,
        "TB/s": nbytes / us / 1e6,
    }


@benchmark()
def test_sorted_sum(token, model_dim, topk, padding):
    dtype = dtypes.bf16
    experts = max(4, topk)
    ids = (torch.arange(token * topk).reshape(token, topk) % experts).to(torch.int32)
    # Sorting keeps one route of a repeated expert ID and gives the others -1.
    ids[-1] = 0
    weights = torch.ones((token, topk), dtype=torch.float32)
    storage = torch.full((token * model_dim + 2 * GUARD,), CANARY, dtype=dtype)
    _, _, sorted_experts, _, output, _, loc_ids = moe_sorting(
        ids,
        weights,
        experts,
        model_dim,
        dtype,
        64,
        output=storage[GUARD:-GUARD].view(token, model_dim),
        output_aux=AUX_SORT_OPUS,
    )
    valid = loc_ids.view(token, topk) >= 0
    assert int((~valid).sum()) == topk - 1, "repeated expert IDs kept extra routes"
    # Specialized Down rows carry padding that holds canaries the sum must skip.
    down_output = torch.full(
        (sorted_experts.numel() * 64, model_dim + padding // 2), CANARY, dtype=dtype
    )
    routes = torch.randn((token, topk, model_dim)).to(dtype)
    down_output[loc_ids.view(token, topk)[valid].long(), :model_dim] = routes[valid]
    reference = torch.zeros((token, model_dim))
    for route in range(topk):
        reference = reference + routes[:, route].float() * valid[:, route, None]
    reduce = sorted_sum(topk, model_dim, padding)

    def run():
        reduce(loc_ids, down_output, output, token)
        return output

    actual, us = run_perftest(run)
    err = checkAllclose(
        reference.to(dtype).float(),
        actual.float(),
        rtol=0,
        atol=0,
        msg=f"sorted_sum padding={padding}",
    )
    assert err == 0, "padded sorted reduction is not exact"
    check_guards([storage])
    flops = token * model_dim * (topk - 1)
    nbytes = token * model_dim * (topk + 1) * output.element_size() + loc_ids.nbytes
    return {
        "us": us,
        "TFLOPS": flops / us / 1e6,
        "TB/s": nbytes / us / 1e6,
        "err": err,
    }


def test_config_strings():
    for encoded in get_tune_space():
        assert Config.from_string(encoded).to_string() == encoded
    # Rows tuned before the Down suffix existed keep loading as the default Down.
    for legacy in ("16_16_16_False", "64_128_128_True", "64_256_128_True"):
        assert Config.from_string(legacy) == Config.from_string(f"{legacy}:default:0")
    # Tiles without the direct-decoding flag keep their meaning.
    assert Config.from_string("16_16_16_False_False:default:0") == Config.from_string(
        "16_16_16_False:default:0"
    )
    for invalid in (
        "64_256_128_True:default",
        "64_256_128_True:default:0:0",
        "64_256_128_Maybe",
        "64_256_True",
        "64_256_128_True:default:128",
        "64_256_128_True:8x2:128",
        "16_16_16_False:8x1_compact:128",
        "16_16_16_False_Maybe:default:0",
        "16_16_16_False_True_False:default:0",
        "16_16_16_True_True:default:0",
        "16_16_16_False_True:1x4_64x256:128",
        "64_128_64_False_True:default:0",
    ):
        try:
            Config.from_string(invalid)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid config {invalid!r}")
    batch1 = _Problem(1, 8, 256, 512, 512, 128, 2, "no")
    assert Config.from_string("16_16_16_False").unsupported_reason(batch1) is None
    for tile in ("32_16_16_False", "16_16_32_False", "16_128_64_False"):
        assert Config.from_string(tile).unsupported_reason(batch1), tile


def test_block_scales():
    """Checkpoint block scales are validated, expanded once and cached."""
    experts, model_dim, inter_dim = 2, 512, 192
    w1 = torch.empty((experts, 2 * inter_dim, model_dim), dtype=dtypes.fp8)
    w2 = torch.empty((experts, model_dim, inter_dim), dtype=dtypes.fp8)
    hidden = torch.empty((4, model_dim), dtype=dtypes.bf16)
    topk_ids = torch.zeros((4, 2), dtype=torch.int32)
    w1_scale = torch.rand((experts, 3, 4)) + 0.5
    w2_scale = torch.rand((experts, 4, 2)) + 0.5

    def problem(s1, s2, quant_type=QuantType.per_128x128):
        return _Problem.from_inputs(hidden, w1, w2, topk_ids, quant_type, s1, s2)

    for quant_type in (QuantType.per_128x128, QuantType.per_1x128):
        block = problem(w1_scale, w2_scale, quant_type)
        assert (block.quant_type, block.scale_k1, block.scale_k2) == ("block", 64, 64)
    for s1, s2 in (
        # The former kernel layout [E, K // g, N] is rejected, not misread.
        (
            flydsl_block_weight_scale(w1_scale, 2 * inter_dim, model_dim, 64),
            flydsl_block_weight_scale(w2_scale, model_dim, inter_dim, 64),
        ),
        (w1_scale.transpose(1, 2), w2_scale),
        (w1_scale, w2_scale[:, :, :1]),
        (w1_scale.bfloat16(), w2_scale),
        (w1_scale, None),
        # 32x32 blocks: only 128 and 64 are supported.
        (torch.rand((experts, 12, 16)) + 0.5, torch.rand((experts, 16, 6)) + 0.5),
    ):
        try:
            problem(s1, s2)
        except ValueError:
            continue
        raise AssertionError(f"accepted scales {_describe(s1)}, {_describe(s2)}")

    # Each kernel row and 64-wide K group reads its checkpoint block.
    kernel = _kernel_block_scale(w2_scale, model_dim, inter_dim, 64)
    rows = torch.arange(model_dim) // 128
    groups = torch.arange(inter_dim // 64) * 64 // 128
    expected = w2_scale[:, rows][:, :, groups].transpose(1, 2)
    assert torch.equal(kernel, expected)
    # Later calls, including on copies of the scales, reuse the expansion.
    assert _kernel_block_scale(w2_scale, model_dim, inter_dim, 64) is kernel
    copied = copy.deepcopy(w2_scale)
    assert torch.equal(_kernel_block_scale(copied, model_dim, inter_dim, 64), kernel)
    # New scales of the same shape get their own expansion.
    other = _kernel_block_scale(2 * w2_scale, model_dim, inter_dim, 64)
    assert torch.equal(other, 2 * expected)
    # A graph capture uses the expansion but does not keep it.
    fresh = w2_scale.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _kernel_block_scale(fresh, model_dim, inter_dim, 64)
    assert not hasattr(fresh, "_flydsl_kernel_scales")

    # 64x64 blocks (vLLM's TP refinement) give 64-wide groups reading their block.
    w1_scale64 = torch.rand((experts, 6, 8)) + 0.5
    w2_scale64 = torch.rand((experts, 8, 3)) + 0.5
    block = problem(w1_scale64, w2_scale64)
    assert (block.scale_k1, block.scale_k2) == (64, 64)
    kernel = _kernel_block_scale(w2_scale64, model_dim, inter_dim, 64)
    expected = w2_scale64[:, torch.arange(model_dim) // 64].transpose(1, 2)
    assert torch.equal(kernel, expected)
    # With inter_dim % 128 == 0, 64-wide Down blocks leave only decode configs.
    w1 = torch.empty((experts, 512, model_dim), dtype=dtypes.fp8)
    w2 = torch.empty((experts, model_dim, 256), dtype=dtypes.fp8)
    block = problem(torch.rand((experts, 8, 8)), torch.rand((experts, 8, 4)))
    assert (block.scale_k1, block.scale_k2) == (64, 64)
    decode = Config.from_string("16_16_16_False:default:0")
    prefill = Config.from_string("64_128_128_True:1x4_64x256:128")
    assert decode.unsupported_reason(block) is None
    assert prefill.unsupported_reason(block)


def route(hidden, experts, topk, routing):
    token = hidden.shape[0]
    if routing in ("random", "padding", "pad_rows"):
        weights, ids = fused_topk(
            hidden, torch.randn((token, experts), dtype=hidden.dtype), topk, True
        )
        if routing == "padding":
            # HIP padding rows: zero weights and expert 0 repeated, which sorting merges.
            weights[-65:] = 0
            ids[-65:, :-1] = 0
        if routing == "pad_rows":
            # Rows past num_valid_rows: -1 ids, zero weights; the last column stays.
            pad = min(65, (token + 1) // 2)
            weights[-pad:, :-1] = 0
            ids[-pad:, :-1] = -1
        return weights, ids
    # "hot": every token picks the same highest experts and the others stay empty.
    ids = torch.arange(experts - topk, experts, dtype=torch.int32)
    weights = torch.rand((token, topk), dtype=torch.float32)
    return (
        weights / weights.sum(dim=-1, keepdim=True),
        ids.expand(token, topk).contiguous(),
    )


def behind_nan_expert(tensor):
    """The tensor stored after a NaN expert, so reading expert -1 shows up."""
    if tensor is None:
        return None
    storage = torch.empty((tensor.shape[0] + 1, *tensor.shape[1:]), dtype=tensor.dtype)
    storage[0].fill_(float("nan"))
    view = storage[1:].copy_(tensor)
    if hasattr(tensor, "is_shuffled"):
        view.is_shuffled = tensor.is_shuffled
    return view


def stale_gateup_output(hidden_states, problem):
    """Gate/Up output that starts as inf, like reused memory."""
    return _gateup_output(hidden_states, problem).fill_(float("inf"))


def block_quant(w, block=128):
    """FP8 weights with 128x128 block scales (partial edge blocks allowed).

    Returns the FP8 weights, their exact FP32 dequantization and the checkpoint
    scales [E, ceil(N / 128), ceil(K / 128)].
    """
    experts, n, k = w.shape
    n_blocks, k_blocks = -(-n // block), -(-k // block)
    padded = torch.nn.functional.pad(
        w.float(), (0, k_blocks * block - k, 0, n_blocks * block - n)
    )
    blocks = padded.view(experts, n_blocks, block, k_blocks, block)
    scale = blocks.abs().amax(dim=(2, 4)) / torch.finfo(dtypes.fp8).max
    expanded = scale.repeat_interleave(block, 1).repeat_interleave(block, 2)
    expanded = expanded[:, :n, :k]
    w_q = (w.float() / expanded).to(dtypes.fp8)
    return w_q, w_q.float() * expanded, scale


def group_quant(x, group):
    """FP8 round trip with one scale per (row, group of K), as the prefill kernels see x."""
    groups = x.float().view(*x.shape[:-1], -1, group)
    scale = groups.abs().amax(dim=-1, keepdim=True) / torch.finfo(dtypes.fp8).max
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    return ((groups / scale).to(dtypes.fp8).float() * scale).view(x.shape)


def block_reference(hidden, w1, w2, weights, ids, activation, swiglu_limit, groups):
    """Torch MoE on dequantized weights. With (Gate/Up, Down) K groups the
    activations take the FP8 round trip of the prefill kernels; decode reads BF16."""
    token, topk = ids.shape
    a1 = group_quant(hidden, groups[0]) if groups else hidden
    stage1 = torch_moe_stage1(
        a1,
        w1,
        w2,
        weights,
        ids,
        dtype=dtypes.bf16,
        activation=activation,
        swiglu_limit=swiglu_limit,
    )
    a2 = group_quant(stage1, groups[1]) if groups else stage1
    return torch_moe_stage2(
        a2.view(token, topk, -1), w1, w2, weights, ids, dtype=dtypes.bf16
    )


def vllm_shard_scale(scale, dim, rank, rows, block=128):
    """Rank `rank`'s 128x128 checkpoint block scales for `rows` rows sharded along
    `dim`, loaded as vLLM does: if a shard starts mid-block, the scales are refined
    to gcd(128, rows) blocks, each taking its checkpoint block's scale, then sliced."""
    refined = math.gcd(block, rows)
    factor = block // refined
    scale = scale.repeat_interleave(factor, 1).repeat_interleave(factor, 2)
    return scale.narrow(dim, rank * rows // refined, rows // refined).contiguous()


def test_tp_block_scales(full_inter, tp=2):
    """TP shards of a Step-5-shaped 128x128-block FP8 checkpoint with vLLM's scales
    match the torch reference on the dequantized shard, through the public entry
    and the tuned aiter.fused_moe dispatch, on an even and an odd rank."""
    torch.manual_seed(0)
    experts, topk, model_dim = 352, 8, 4096
    inter = full_inter // tp
    block = math.gcd(128, inter)
    act = ActivationType.SwigluStep
    gate, up = (
        block_quant(torch.randn((experts, full_inter, model_dim), dtype=dtypes.bf16))
        for _ in range(2)
    )
    down = block_quant(torch.randn((experts, model_dim, full_inter), dtype=dtypes.bf16))
    configs = {
        row[0]: row[-1]
        for row in shipped_rows()
        if row[1:6] == (model_dim, inter, experts, topk, "block")
    }
    dispatch = get_cu_num() == 304
    for rank in range(tp):
        rows = slice(rank * inter, (rank + 1) * inter)
        w1_q, w1_ref = (
            torch.cat((gate[i][:, rows], up[i][:, rows]), 1) for i in (0, 1)
        )
        w2_q, w2_ref = down[0][:, :, rows].contiguous(), down[1][:, :, rows]
        w1_scale = torch.cat(
            [vllm_shard_scale(w[2], 1, rank, inter) for w in (gate, up)], 1
        )
        w2_scale = vllm_shard_scale(down[2], 2, rank, inter)
        assert w1_scale.shape == (experts, 2 * inter // block, model_dim // block)
        assert w2_scale.shape == (experts, model_dim // block, inter // block)
        w1 = shuffle_weight(w1_q, layout=(16, 16))
        w2 = shuffle_weight(w2_q, layout=(16, 16))
        for token in (1, 4, 64, 512):
            config = configs[token]
            hidden = torch.randn((token, model_dim), dtype=dtypes.bf16)
            weights, ids = fused_topk(
                hidden, torch.randn((token, experts), dtype=hidden.dtype), topk, True
            )
            reference = block_reference(
                hidden,
                w1_ref,
                w2_ref,
                weights,
                ids,
                act,
                SWIGLU_STEP_LIMIT,
                (
                    block_scale_groups(inter, block)
                    if Config.from_string(config).use_prefill
                    else None
                ),
            )
            args = (hidden, w1, w2, weights, ids, act, QuantType.per_128x128)
            msg = f"TP{tp} rank {rank} {block}x{block} token {token} {config}"
            output = run_flydsl_moe_gfx942(
                *args, w1_scale, w2_scale, None, None, 0, config, SWIGLU_STEP_LIMIT
            )
            check_output(reference, output, msg)
            if not dispatch:
                continue
            with patch(
                "aiter.ops.flydsl.fused_moe_gfx942.run_flydsl_moe_gfx942",
                wraps=run_flydsl_moe_gfx942,
            ) as spy:
                output = fused_moe(
                    *args[:5],
                    activation=act,
                    quant_type=QuantType.per_128x128,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    swiglu_limit=SWIGLU_STEP_LIMIT,
                )
            assert spy.call_args.args[12] == config, (msg, spy.call_args)
            check_output(reference, output, f"fused_moe {msg}")


@functools.lru_cache(maxsize=1)
def make_case(
    token,
    model_dim,
    inter_dim,
    experts,
    topk,
    quant,
    activation,
    routing,
    quantize_activations,
):
    """Inputs and torch reference as test_moe_2stage.py builds them."""
    torch.manual_seed(0)
    q_type, act_type = QUANTS[quant], ACTIVATIONS[activation]
    q_dtype = None if q_type == QuantType.No else dtypes.fp8
    torch_quant = aiter.get_torch_quant(q_type)
    swiglu_limit = SWIGLU_STEP_LIMIT if activation == "swiglustep" else None

    def quantize_activation(x):
        # Decode and batch-1 kernels read BF16 activations even with FP8 weights.
        if quantize_activations:
            return torch_quant(x, quant_dtype=q_dtype)
        return x, torch.ones(1, dtype=torch.float32)

    hidden = torch.randn((token, model_dim), dtype=dtypes.bf16)
    w1 = torch.randn((experts, 2 * inter_dim, model_dim), dtype=dtypes.bf16)
    w2 = torch.randn((experts, model_dim, inter_dim), dtype=dtypes.bf16)
    topk_weights, topk_ids = route(hidden, experts, topk, routing)
    if q_type == QuantType.per_128x128:
        w1_q, w1_ref, w1_scale = block_quant(w1)
        w2_q, w2_ref, w2_scale = block_quant(w2)
        reference = block_reference(
            hidden,
            w1_ref,
            w2_ref,
            topk_weights,
            topk_ids,
            act_type,
            swiglu_limit,
            block_scale_groups(inter_dim) if quantize_activations else None,
        )
    else:
        if q_type == QuantType.per_Tensor:
            w1_q, w1_scale = aiter.pertoken_quant(
                w1.view(experts, -1), quant_dtype=q_dtype
            )
            w2_q, w2_scale = aiter.pertoken_quant(
                w2.view(experts, -1), quant_dtype=q_dtype
            )
            w1_q, w2_q = w1_q.view(w1.shape), w2_q.view(w2.shape)
        else:
            w1_q, w1_scale = torch_quant(w1, quant_dtype=q_dtype)
            w2_q, w2_scale = torch_quant(w2, quant_dtype=q_dtype)
        a1, a1_scale = quantize_activation(hidden)
        stage1 = torch_moe_stage1(
            a1,
            w1_q,
            w2_q,
            topk_weights,
            topk_ids,
            dtype=dtypes.bf16,
            activation=act_type,
            quant_type=q_type,
            a1_scale=a1_scale,
            w1_scale=w1_scale,
            swiglu_limit=swiglu_limit,
        )
        a2, a2_scale = quantize_activation(stage1)
        reference = torch_moe_stage2(
            a2.view(token, topk, -1),
            w1_q,
            w2_q,
            topk_weights,
            topk_ids,
            dtype=dtypes.bf16,
            quant_type=q_type,
            w2_scale=w2_scale,
            a2_scale=a2_scale,
        )
    request = FusedMoeRequest(
        hidden_states=hidden,
        w1=behind_nan_expert(shuffle_weight(w1_q, layout=(16, 16))),
        w2=behind_nan_expert(shuffle_weight(w2_q, layout=(16, 16))),
        topk_weight=topk_weights,
        topk_ids=topk_ids,
        activation=act_type,
        quant_type=q_type,
        w1_scale=behind_nan_expert(w1_scale),
        w2_scale=behind_nan_expert(w2_scale),
        swiglu_limit=swiglu_limit,
    )
    # Holds the first FlyDSL output of the case for the cross-path check.
    return request, reference, {}


class GuardedTaskBuffers:
    """Replaces allocate_task_buffers and keeps the last allocation for checks."""

    def __call__(self, sorted_expert_ids, experts, cu_count):
        self.metadata, self.cu_count = sorted_expert_ids, cu_count
        self.storages, self.buffers = guarded_task_buffers(
            sorted_expert_ids, experts, cu_count
        )
        return self.buffers


def token_rms(x):
    return x.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)


def token_error(expected, actual):
    """Largest per-token relative L2 error."""
    x, y = expected.double(), actual.double()
    return float(((x - y).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-6)).max())


def check_output(reference, output, msg):
    """Assert every accuracy check; returns the checkAllclose mismatch ratio."""
    assert bool(torch.isfinite(output).all()), f"{msg}: non-finite output"
    # Errors in units of each token's reference RMS, so one tolerance fits all shapes.
    rms = token_rms(reference)
    err = checkAllclose(
        reference.float() / rms,
        output.float() / rms,
        rtol=0,
        atol=ELEMENT_ERR_LIMIT,
        msg=msg,
    )
    x, y = reference.double(), output.double()
    logits_diff = float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())
    token_err = token_error(reference, output)
    assert err == 0, f"{msg}: {err:.3%} of elements beyond {ELEMENT_ERR_LIMIT} RMS"
    assert logits_diff < LOGITS_DIFF_LIMIT, f"{msg}: logits_diff={logits_diff}"
    assert token_err < TOKEN_ERR_LIMIT, f"{msg}: worst token error {token_err}"
    return err


def run_compact(impl, request, experts, cu_count):
    """Run with guarded task buffers and check the task table built for cu_count."""
    buffers = GuardedTaskBuffers()
    with patch.object(compact, "allocate_task_buffers", buffers):
        output = impl(request)
    assert buffers.cu_count == cu_count
    sizes = torch.bincount(request.topk_ids.flatten().long(), minlength=experts)
    blocks = buffers.metadata[: int(((sizes + 63) // 64).sum())].tolist()
    full, tail = check_tasks(blocks, experts, cu_count, *buffers.buffers)
    check_guards(buffers.storages)
    return output, f"{full}/{tail}"


def rms_error(expected, actual, rms):
    """Largest elementwise difference in units of the token's reference RMS."""
    return float(((actual.float() - expected.float()).abs() / rms).max())


def check_moe(
    token, model_dim, inter_dim, experts, topk, quant, activation, routing, config
):
    """Run one config through the registered backend and assert its accuracy."""
    parsed = Config.from_string(config)
    request, reference, first_output = make_case(
        token,
        model_dim,
        inter_dim,
        experts,
        topk,
        quant,
        activation,
        routing,
        parsed.use_prefill,
    )
    impl = resolve_fused_moe_impl(make_fused_moe_impl_kernel_name(IMPL, config))
    rms = token_rms(reference)
    tasks = "-"
    with patch("aiter.ops.flydsl.fused_moe_gfx942._gateup_output", stale_gateup_output):
        if parsed.down_path == "8x1_compact":
            device_cu = get_cu_num()
            output, tasks = run_compact(impl, request, experts, device_cu)
            # The cached launcher must stay in bounds and correct for the other CU count.
            for cu_count in CU_COUNTS:
                if cu_count != device_cu:
                    with patch(
                        "aiter.ops.flydsl.fused_moe_gfx942.get_cu_num",
                        return_value=cu_count,
                    ):
                        other, _ = run_compact(impl, request, experts, cu_count)
                    diff = rms_error(output, other, rms)
                    assert (
                        diff < PATH_DIFF_LIMIT
                    ), f"{config} at {cu_count} CUs: {diff=}"
        else:
            output = impl(request)
    err = check_output(reference, output, config)
    # Decode kernels accumulate routes with atomics, so only prefill paths are compared.
    if parsed.use_prefill:
        first = first_output.setdefault("output", output.clone())
        path_diff = rms_error(first, output, rms)
        assert (
            path_diff < PATH_DIFF_LIMIT
        ), f"{config}: {path_diff=} from the first path"
    return impl, request, tasks, err


@benchmark()
def test_moe(token, model_dim, inter_dim, experts, topk, quant, activation, config):
    impl, request, tasks, err = check_moe(
        token, model_dim, inter_dim, experts, topk, quant, activation, "random", config
    )
    _, us = run_perftest(impl, request, num_iters=10, num_warmup=2)
    flops = 6 * token * topk * model_dim * inter_dim
    # Input and output activations plus both weight tensors.
    nbytes = 2 * request.hidden_states.nbytes + request.w1.nbytes + request.w2.nbytes
    return {
        "full/tail tasks": tasks,
        "us": us,
        "TFLOPS": flops / us / 1e6,
        "TB/s": nbytes / us / 1e6,
        "err": err,
    }


def shipped_rows():
    """Every shipped FlyDSL row as test_moe arguments, grouped by shape."""
    df = pd.read_csv(AITER_CONFIGS.AITER_CONFIG_FMOE_FILE)
    df = df[df["kernelName1"].astype(str).str.startswith(f"impl__{IMPL}__")]
    if "_tag" in df:
        df = df[df["_tag"].fillna("").astype(str).str.strip() == ""]
    quants = {
        "No": "no",
        "per_Tensor": "per_tensor",
        "per_Token": "ptpc",
        "per_1x128": "block",
    }
    rows = set()
    for _, row in df.iterrows():
        shape = (
            int(row["model_dim"]),
            int(row["inter_dim"]),
            int(row["expert"]),
            int(row["topk"]),
            quants[str(row["q_type"]).split(".")[-1]],
            str(row["act_type"]).split(".")[-1].lower(),
        )
        rows.add((shape, int(row["token"]), str(row["kernelName1"]).split("__", 2)[2]))
    return [(token, *shape, config) for shape, token, config in sorted(rows)]


def cross_config_cases(rows):
    """All shipped prefill configs of a shape on the same inputs, so outputs compare."""
    configs = {}
    for _, *shape, config in rows:
        if Config.from_string(config).use_prefill:
            configs.setdefault(tuple(shape), set()).add(config)
    return [
        (16384, *shape, "random", config)
        for shape, shape_configs in sorted(configs.items())
        if len(shape_configs) > 1
        for config in sorted(shape_configs)
    ]


# Functional checks only, not timed:
# (token, model_dim, inter_dim, experts, topk, quant, activation, routing, config)
EDGE_CASES = [
    # The 2048-expert limit of the compact task table, with all or 8 experts active.
    *(
        (
            4096,
            512,
            256,
            2048,
            8,
            "ptpc",
            "silu",
            routing,
            f"64_256_128_True:{path}:128",
        )
        for routing in ("random", "hot")
        for path in ("8x1_compact", "1x4_64x256")
    ),
    # One expert with 64 full tasks: kept on 80 CUs, split into tails on 304.
    *(
        (16384, 1024, 384, 1, 1, "per_tensor", "silu", "hot", config)
        for config in (
            "64_256_128_True:8x1_compact:128",
            "64_256_128_True:1x4_64x256:128",
            "64_256_128_True:default:0",
        )
    ),
    # Every row padding of the specialized Down paths, reduced with the same stride.
    *(
        (
            16384,
            2048,
            512,
            257,
            9,
            "ptpc",
            "silu",
            "random",
            f"64_256_128_True:{path}:{padding}",
        )
        for path in ("8x1_compact", "1x4_64x256")
        for padding in (0, 32, 64)
    ),
    # Padding rows whose merged routes reach the reduction as -1.
    *(
        (2048, 2560, 320, 513, 11, quant, "silu", "padding", config)
        for quant, config in (
            ("no", "64_128_128_True:default:0"),
            ("ptpc", "64_128_256_True:1x4_64x256:128"),
        )
    ),
    # Gate/Up skips -1 pad rows, but the per-tensor absmax reads every row.
    (
        2048,
        2560,
        320,
        513,
        11,
        "per_tensor",
        "silu",
        "pad_rows",
        "64_128_256_True:1x4_64x256:128",
    ),
    # Direct decoding reads -1 pad rows from topk_ids without sorting.
    (2, 2560, 320, 513, 11, "ptpc", "silu", "pad_rows", "16_16_16_False_True"),
    # Block-scaled FP8 with SwiGLU-step on every algorithm. inter_dim 192 ends in a
    # partial 128-wide K block of w2 (64-wide Down groups); 256 uses 128-wide groups.
    *(
        (token, 1024, inter_dim, 32, 8, "block", "swiglustep", "random", config)
        for inter_dim in (192, 256)
        for token, config in (
            (1, "16_16_16_False:default:0"),
            (5, "16_16_16_False_True:default:0"),
            (64, "16_16_16_False:default:0"),
            (64, "16_128_64_False:default:0"),
            (2048, "64_128_128_True:1x4_64x256:128"),
        )
    ),
    # Hot routing (all tokens same expert) on one decode config per inter_dim.
    *(
        (
            64,
            1024,
            inter_dim,
            32,
            8,
            "block",
            "swiglustep",
            "hot",
            "16_16_16_False:default:0",
        )
        for inter_dim in (192, 256)
    ),
    # Step-5 shape (model_dim 4096, inter_dim 192 partial K block) at CI-friendly
    # sizes; the 2048-token prefill above already covers the full kernel path.
    (8, 4096, 192, 64, 8, "block", "swiglustep", "random", "16_16_16_False:default:0"),
    (
        2048,
        4096,
        192,
        64,
        8,
        "block",
        "swiglustep",
        "random",
        "64_128_128_True:1x4_64x256:128",
    ),
]


def summarize(name, rows):
    aiter.logger.info(
        "%s summary (markdown):\n%s", name, pd.DataFrame(rows).to_markdown(index=False)
    )


def main():
    if get_gfx() != "gfx942":
        aiter.logger.warning(
            "gfx942 FlyDSL whole-MoE backend unsupported on %s; skipping", get_gfx()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=None,
        help="Only time the shipped rows with these token counts (default: every row).",
    )
    args = parser.parse_args()

    test_config_strings()
    test_block_scales()
    summarize(
        "Compact task table",
        [
            test_task_table(scenario, cu_count, cu_count)
            for cu_count in CU_COUNTS
            for scenario in task_scenarios(cu_count)
        ]
        # Buffers sized for 80 CUs under a 304-CU split: the device guard caps it.
        + [test_task_table("cu_dependent", *CU_COUNTS)],
    )
    summarize(
        "Padded sorted reduction",
        [
            test_sorted_sum(token, model_dim, topk, padding)
            for token, (model_dim, topk), padding in itertools.product(
                (65, 2048),
                ((2048, 9), (2560, 11), (4096, 8), (6144, 4)),
                (0, 32, 64, 128),
            )
        ],
    )
    rows = shipped_rows()
    for case in cross_config_cases(rows) + EDGE_CASES:
        check_moe(*case)
    # vLLM TP shards: inter 384 -> 192 refines scales to 64x64; 768 -> 384 keeps 128x128.
    for full_inter in (384, 768):
        test_tp_block_scales(full_inter)
    timed = [row for row in rows if not args.batch or row[0] in args.batch]
    summarize("Whole MoE", [test_moe(*row) for row in timed])


if __name__ == "__main__":
    main()
