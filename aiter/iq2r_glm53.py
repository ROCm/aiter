# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3 IQ2R MoE on the packed layout (TP4 and TP8, gfx950).

The packed layout is a byte-exact relayout of the canonical IQ2R records, so
decoded FP8 weights are unchanged:

* codebook signs move into the record sign bytes (codebook bit 7 cleared);
* the four 9-bit indices of an atom pack as low32 + high4;
* gate/up record signs transpose into four 8-bit planes, and the gate/up
  records are regrouped into quads of four 16-column blocks.

Every transform is per record, so packing a full checkpoint and then taking
a tensor-parallel shard (``iq2r_glm53_slice_gate``, ``iq2r_slice_output_auxiliary``,
``iq2r_slice_input_data``) equals packing that shard.

Launch choices per token count come from ``aiter/configs/iq2r_glm53_tuned.csv``
(override with ``AITER_CONFIG_IQ2R_GLM53``); shapes missing from the table use
the defaults in :func:`_default_config`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import pandas as pd
import torch
from torch import Tensor

from aiter import logger

from .jit.core import AITER_CONFIGS, AITER_LOG_TUNED_CONFIG
from .jit.utils.chip_info import get_cu_num
from .jit.utils.chip_info import get_gfx_runtime as get_gfx
from .ops.iq2r import (
    iq2r_glm53_down_out,
    iq2r_glm53_down_reduce_out,
    iq2r_glm53_down_route9_out,
    iq2r_glm53_gate_m1_out,
    iq2r_glm53_gate_out,
    iq2r_glm53_route_reduce_out,
    iq2r_glm53_sort_out,
    iq2r_glm53_sort_quant_out,
    iq2r_route_direct_gather_quant_out,
    iq2r_route_gather_quant_out,
)

HIDDEN = 6144
EXPERTS = 257
TOPK = 9
MAX_DECODE_TOKENS = 256
MAX_CHUNK_TOKENS = 4096
INTERMEDIATE_SIZES = (256, 512)  # TP8, TP4

_RECORD_BYTES = 3584
_QUAD_BYTES = 48 * 2304

GATE_KERNELS = {"decode": 0, "nobarrier": 1, "prefill": 2}
DOWN_KERNELS = {"packed": 0, "ordered": 1}


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------


def _geometry(data: Tensor, logical_k: int) -> tuple[int, int]:
    experts = data.shape[0]
    k_tiles = logical_k // 128
    if data.shape[1] % (k_tiles * _RECORD_BYTES):
        raise ValueError("unexpected IQ2R record layout")
    return experts, k_tiles


def _atoms(src: Tensor, dst: Tensor, experts: int, k_tiles: int):
    """Yield (record, dst_record, third, dst_third, atom) views per 8-byte atom."""
    s = src.reshape(experts, -1, k_tiles, _RECORD_BYTES)
    d = dst.reshape_as(s)
    for triplet in range(2):
        offset = triplet * 1792
        pair = s[..., offset : offset + 1024].reshape(experts, -1, k_tiles, 64, 16)
        third = s[..., offset + 1024 : offset + 1792].reshape(
            experts, -1, k_tiles, 64, 12
        )
        dst_pair = d[..., offset : offset + 1024].reshape_as(pair)
        dst_third = d[..., offset + 1024 : offset + 1792].reshape_as(third)
        for atom in range(3):
            record = pair[..., atom * 8 : atom * 8 + 8] if atom < 2 else third[..., :8]
            out = (
                dst_pair[..., atom * 8 : atom * 8 + 8]
                if atom < 2
                else dst_third[..., :8]
            )
            yield record, out, third, dst_third, atom


def _normalize_codebook_signs(
    data: Tensor, auxiliary: Tensor, logical_k: int
) -> tuple[Tensor, Tensor]:
    experts, k_tiles = _geometry(data, logical_k)
    result = data.clone()
    aux = auxiliary.clone()
    book = auxiliary[:, :4096].reshape(experts, 512, 8)
    signs = torch.zeros((experts, 512), device=data.device, dtype=torch.int64)
    for byte in range(8):
        signs |= (book[:, :, byte].to(torch.int64) >> 7) << byte
    for record, out, third, _, atom in _atoms(data, result, experts, k_tiles):
        metadata = third[..., 8 + atom].to(torch.int64)
        for word in range(4):
            index = record[..., word].to(torch.int64) + (((metadata >> word) & 1) << 8)
            flip = signs.gather(1, index.flatten(1)).reshape_as(index).to(torch.uint8)
            out[..., 4 + word].copy_(record[..., 4 + word] ^ flip)
    aux[:, :4096] &= 127
    return result, aux


def _pack_indices(data: Tensor, logical_k: int) -> Tensor:
    experts, k_tiles = _geometry(data, logical_k)
    result = data.clone()
    for record, out, third, dst_third, atom in _atoms(data, result, experts, k_tiles):
        metadata = third[..., 8 + atom].to(torch.int64)
        packed = torch.zeros_like(metadata)
        for word in range(4):
            index = record[..., word].to(torch.int64) | (((metadata >> word) & 1) << 8)
            packed |= index << (word * 9)
        for byte in range(4):
            out[..., byte].copy_(((packed >> (8 * byte)) & 255).to(torch.uint8))
        dst_third[..., 8 + atom].copy_(
            ((metadata & 240) | ((packed >> 32) & 15)).to(torch.uint8)
        )
    return result


def _transpose_signs(data: Tensor, logical_k: int) -> Tensor:
    experts, k_tiles = _geometry(data, logical_k)
    result = data.clone()
    for record, out, _, _, _ in _atoms(data, result, experts, k_tiles):
        old = sum(record[..., 4 + j].to(torch.int64) << (8 * j) for j in range(4))
        new = torch.zeros_like(old)
        for bit in range(32):
            new |= ((old >> bit) & 1) << ((bit % 4) * 8 + bit // 4)
        for j in range(4):
            out[..., 4 + j].copy_(((new >> (8 * j)) & 255).to(torch.uint8))
    return result


def _quad_gate(data: Tensor, logical_n: int) -> Tensor:
    """Regroup K=6144 gate/up records into [experts, N/64, 48, 2304]."""
    groups = data.shape[1] // (48 * _RECORD_BYTES)
    blocks = logical_n // 16
    if logical_n % 64 or (blocks + 5) // 6 != groups:
        raise ValueError("gate/up N must be a multiple of 64 matching the data")
    experts = data.shape[0]
    source = data.reshape(experts, groups, 48, _RECORD_BYTES)
    target = torch.empty(
        (experts, blocks // 4, 48, 2304), dtype=torch.uint8, device=data.device
    )
    for block in range(blocks):
        atom = block % 3
        offset = (block % 6) // 3 * 1792
        third = source[:, block // 6, :, offset + 1024 : offset + 1792].reshape(
            experts, 48, 64, 12
        )
        if atom < 2:
            pair = source[:, block // 6, :, offset : offset + 1024].reshape(
                experts, 48, 64, 16
            )
            record = pair[..., atom * 8 : atom * 8 + 8]
        else:
            record = third[..., :8]
        slot = block % 4
        pair_base = (slot // 2) * 1024
        pair_out = target[:, block // 4, :, pair_base : pair_base + 1024].view(
            experts, 48, 64, 16
        )
        pair_out[..., slot % 2 * 8 : slot % 2 * 8 + 8].copy_(record)
        metadata = target[:, block // 4, :, 2048:2304].view(experts, 48, 64, 4)
        metadata[..., slot].copy_(third[..., 8 + atom])
    return target.flatten(1).contiguous()


@torch.no_grad()
def iq2r_glm53_pack(
    gate_up_data: Tensor,
    gate_up_auxiliary: Tensor,
    down_data: Tensor,
    down_auxiliary: Tensor,
    *,
    intermediate_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Relayout one GLM-5.3 MoE layer (full checkpoint or a TP shard).

    Returns ``(gate_quad, gate_auxiliary, down_data, down_auxiliary)``.
    """
    gate, gate_aux = _normalize_codebook_signs(gate_up_data, gate_up_auxiliary, HIDDEN)
    gate = _transpose_signs(_pack_indices(gate, HIDDEN), HIDDEN)
    quad = _quad_gate(gate, 2 * intermediate_size)
    del gate
    down, down_aux = _normalize_codebook_signs(
        down_data, down_auxiliary, intermediate_size
    )
    return quad, gate_aux, _pack_indices(down, intermediate_size), down_aux


def iq2r_glm53_gate_bytes(logical_n: int) -> int:
    """Bytes per expert of a packed GLM-5.3 gate/up matrix (K=6144)."""
    if logical_n % 64:
        raise ValueError("packed gate/up N must be a multiple of 64")
    return logical_n // 64 * _QUAD_BYTES


def iq2r_glm53_slice_gate(gate: Tensor, start: int, length: int) -> Tensor:
    """Output-column slice [start, start+length) of a packed gate/up matrix.

    Both bounds must be multiples of 64 so the slice consists of whole quads.
    """
    if start % 64 or length % 64 or gate.shape[1] % _QUAD_BYTES:
        raise ValueError("packed gate/up slices must be aligned to 64 columns")
    quads = gate.view(gate.shape[0], -1, _QUAD_BYTES)
    if (start + length) // 64 > quads.shape[1]:
        raise ValueError("packed gate/up slice out of range")
    picked = quads[:, start // 64 : (start + length) // 64]
    return picked.reshape(gate.shape[0], -1).clone()


# --------------------------------------------------------------------------
# Tuned dispatch
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IQ2RGlm53Config:
    gate_kernel: str
    gate_grid: int
    down_kernel: str
    down_grid: int
    down_chunks: int


def _default_config(tokens: int, intermediate_size: int) -> IQ2RGlm53Config:
    if tokens > MAX_DECODE_TOKENS:
        return IQ2RGlm53Config("prefill", 2, "prefill", 0, 1 if tokens <= 2560 else 2)
    down = "route9" if tokens in (1, 2, 4) else "packed"
    return IQ2RGlm53Config("decode", 2, down, 4, 1)


_TUNED: dict[str, dict] = {}


def _tuned_table(tuned_file: str) -> dict:
    if tuned_file not in _TUNED:
        table = pd.read_csv(tuned_file).drop_duplicates()
        _TUNED[tuned_file] = table.set_index(
            ["gfx", "cu_num", "token", "inter_dim"]
        ).to_dict("index")
    return _TUNED[tuned_file]


@functools.lru_cache(maxsize=1024)
def iq2r_glm53_config(tokens: int, intermediate_size: int) -> IQ2RGlm53Config:
    tuned_file = AITER_CONFIGS.AITER_CONFIG_IQ2R_GLM53_FILE
    row = _tuned_table(tuned_file).get(
        (get_gfx(), get_cu_num(), tokens, intermediate_size)
    )
    if row is None:
        return _default_config(tokens, intermediate_size)
    if AITER_LOG_TUNED_CONFIG:
        logger.info(
            f"IQ2R GLM-5.3 M={tokens} I={intermediate_size}: tuned {row} "
            f"from {tuned_file}"
        )
    return IQ2RGlm53Config(
        str(row["gate_kernel"]),
        int(row["gate_grid"]),
        str(row["down_kernel"]),
        int(row["down_grid"]),
        int(row["down_chunks"]),
    )


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def _task_capacity(routes: int, rows: int) -> int:
    return (routes + rows - 1) // rows + min(routes, EXPERTS + 1)


@dataclass(slots=True)
class IQ2RGlm53Workspace:
    """Fixed-capacity buffers shared by every MoE layer on one device."""

    max_tokens: int
    intermediate_size: int
    route_input_fp8: Tensor
    route_input_scales: Tensor
    intermediate_fp8: Tensor
    intermediate_scales: Tensor
    route_output: Tensor
    sorted_expert_ids: Tensor
    gather_indices: Tensor
    scatter_indices: Tensor
    tasks: Tensor
    task_count: Tensor
    gate_tasks: Tensor
    gate_task_count: Tensor
    sort_scratch: Tensor
    token_identity: Tensor

    @classmethod
    def allocate(
        cls, max_tokens: int, intermediate_size: int, *, device: torch.device | str
    ) -> IQ2RGlm53Workspace:
        if intermediate_size not in INTERMEDIATE_SIZES:
            raise ValueError(
                "GLM-5.3 IQ2R supports intermediate 256 (TP8) or 512 (TP4)"
            )
        # Longer batches run in MAX_CHUNK_TOKENS pieces.
        max_tokens = min(max(max_tokens, 1), MAX_CHUNK_TOKENS)
        routes = max_tokens * TOPK
        rows = max(max_tokens, TOPK)  # M1 quantizes each of its nine routes
        decode_routes = min(max_tokens, MAX_DECODE_TOKENS) * TOPK
        i32 = {"dtype": torch.int32, "device": device}
        u8 = {"dtype": torch.uint8, "device": device}
        fp8 = {"dtype": torch.float8_e4m3fn, "device": device}
        return cls(
            max_tokens=max_tokens,
            intermediate_size=intermediate_size,
            route_input_fp8=torch.empty((rows, HIDDEN), **fp8),
            route_input_scales=torch.empty((rows, HIDDEN // 32), **u8),
            intermediate_fp8=torch.empty((routes, intermediate_size), **fp8),
            intermediate_scales=torch.empty((routes, intermediate_size // 32), **u8),
            route_output=torch.empty(
                (routes, HIDDEN), dtype=torch.bfloat16, device=device
            ),
            sorted_expert_ids=torch.empty((routes,), **i32),
            gather_indices=torch.empty((routes,), **i32),
            scatter_indices=torch.empty((routes,), **i32),
            tasks=torch.empty(
                (max(_task_capacity(routes, 64), _task_capacity(decode_routes, 32)), 3),
                **i32,
            ),
            task_count=torch.empty((1,), **i32),
            gate_tasks=torch.empty((_task_capacity(decode_routes, 16), 3), **i32),
            gate_task_count=torch.empty((1,), **i32),
            sort_scratch=torch.empty(((routes + 255) // 256 * 512 + 1024,), **i32),
            token_identity=torch.arange(max_tokens, **i32),
        )


def iq2r_glm53_moe_out(
    hidden_states: Tensor,
    gate_up_data: Tensor,
    gate_up_auxiliary: Tensor,
    down_data: Tensor,
    down_auxiliary: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    output: Tensor,
    workspace: IQ2RGlm53Workspace,
    config: IQ2RGlm53Config | None = None,
) -> None:
    """Routed GLM-5.3 MoE on packed weights: ``output = sum_k w_k * expert_k(x)``.

    ``hidden_states`` is BF16 [tokens, 6144]; ``topk_ids``/``topk_weights``
    are int32/float32 [tokens, 9] (eight routed experts plus the fused shared
    expert 256); the weights come from :func:`iq2r_glm53_pack`. ``config``
    overrides the tuned launch choice (used by the tuner).
    """
    tokens = hidden_states.shape[0]
    if tokens > workspace.max_tokens:
        for begin in range(0, tokens, workspace.max_tokens):
            rows = slice(begin, min(begin + workspace.max_tokens, tokens))
            iq2r_glm53_moe_out(
                hidden_states[rows],
                gate_up_data,
                gate_up_auxiliary,
                down_data,
                down_auxiliary,
                topk_weights[rows],
                topk_ids[rows],
                output[rows],
                workspace,
                config,
            )
        return
    if tokens == 0:
        return
    if topk_ids.shape != (tokens, TOPK) or topk_weights.shape != (tokens, TOPK):
        raise ValueError("GLM-5.3 IQ2R expects top-9 routing [tokens, 9]")
    if topk_weights.dtype != torch.float32 or not topk_weights.is_contiguous():
        raise ValueError("topk_weights must be contiguous float32")
    if config is None:
        config = iq2r_glm53_config(tokens, workspace.intermediate_size)
    routes = tokens * TOPK
    ids = topk_ids.reshape(-1)
    sorted_ids = workspace.sorted_expert_ids[:routes]
    gather = workspace.gather_indices[:routes]
    scatter = workspace.scatter_indices[:routes]
    tasks = workspace.tasks
    count = workspace.task_count
    intermediate = workspace.intermediate_fp8[:routes]
    intermediate_scales = workspace.intermediate_scales[:routes]
    quant_rows = TOPK if tokens == 1 else tokens
    quant = workspace.route_input_fp8[:quant_rows]
    quant_scales = workspace.route_input_scales[:quant_rows]

    if tokens == 1:
        # Nine one-row tasks; each route gets its own copy of the quantized row.
        tasks = tasks[:routes]
        iq2r_route_direct_gather_quant_out(
            hidden_states, ids, sorted_ids, gather, scatter, tasks, count,
            quant, quant_scales, topk=TOPK, expert_count=EXPERTS,
        )  # fmt: skip
        iq2r_glm53_gate_m1_out(
            quant, quant_scales, gate_up_data, gate_up_auxiliary, tasks, count,
            intermediate, intermediate_scales,
        )  # fmt: skip
    elif tokens <= MAX_DECODE_TOKENS:
        # One launch sorts routes into M32 down / M16 gate tasks and quantizes
        # each token row once; the gate gathers rows through ``gather``.
        tasks = tasks[: _task_capacity(routes, 32)]
        gate_tasks = workspace.gate_tasks[: _task_capacity(routes, 16)]
        iq2r_glm53_sort_quant_out(
            hidden_states, ids, sorted_ids, gather, scatter, tasks, count,
            quant, quant_scales, gate_tasks, workspace.gate_task_count,
        )  # fmt: skip
        iq2r_glm53_gate_out(
            quant, quant_scales, gate_up_data, gate_up_auxiliary, gate_tasks,
            workspace.gate_task_count, gather, intermediate, intermediate_scales,
            GATE_KERNELS[config.gate_kernel], config.gate_grid,
        )  # fmt: skip
    else:
        tasks = tasks[: _task_capacity(routes, 64)]
        iq2r_glm53_sort_out(
            ids, sorted_ids, gather, scatter, tasks, count, workspace.sort_scratch
        )
        iq2r_route_gather_quant_out(
            hidden_states, workspace.token_identity[:tokens], quant, quant_scales,
            topk=1,
        )  # fmt: skip
        iq2r_glm53_gate_out(
            quant, quant_scales, gate_up_data, gate_up_auxiliary, tasks, count,
            gather, intermediate, intermediate_scales, GATE_KERNELS["prefill"],
            config.gate_grid,
        )  # fmt: skip
        iq2r_glm53_down_reduce_out(
            intermediate, intermediate_scales, down_data, down_auxiliary, tasks,
            count, workspace.route_output[:routes], scatter, topk_weights, output,
            config.down_chunks,
        )  # fmt: skip
        return

    if config.down_kernel == "route9":
        # Token-owned down projection with the weighted top-9 sum fused in.
        iq2r_glm53_down_route9_out(
            intermediate, intermediate_scales, down_data, down_auxiliary, ids,
            scatter, topk_weights, output,
        )  # fmt: skip
        return
    route_output = workspace.route_output[:routes]
    iq2r_glm53_down_out(
        intermediate, intermediate_scales, down_data, down_auxiliary, tasks, count,
        route_output, DOWN_KERNELS[config.down_kernel], config.down_grid,
    )  # fmt: skip
    iq2r_glm53_route_reduce_out(route_output, topk_weights, scatter, output)


__all__ = [
    "IQ2RGlm53Config",
    "IQ2RGlm53Workspace",
    "iq2r_glm53_config",
    "iq2r_glm53_gate_bytes",
    "iq2r_glm53_moe_out",
    "iq2r_glm53_pack",
    "iq2r_glm53_slice_gate",
]
