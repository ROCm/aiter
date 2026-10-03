# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""BF16 SwiGLU/SiLU-gated fixed-top-k MoE backward prototype.

This is deliberately a functional prototype rather than the final public
autograd API. It reuses Opus sorting metadata for the route GEMMs and keeps a
compact route list for the two weight-gradient GEMMs.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_down_bwd_fused_kernel(
    dout_ptr,
    z_ptr,
    w2_ptr,
    score_ptr,
    sorted_token_ids_ptr,
    sorted_expert_ids_ptr,
    num_valid_ids_ptr,
    dz_ptr,
    a_scaled_ptr,
    ds_partial_ptr,
    token_num,
    D: tl.constexpr,
    I: tl.constexpr,
    TOPK: tl.constexpr,
    DS_PARTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fuses dO@W2, dS partials, SiLU backward, and S*A."""

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_base = pid_m * BLOCK_M
    num_valid_rows = tl.load(num_valid_ids_ptr)
    if row_base >= num_valid_rows:
        return

    rows = row_base + tl.arange(0, BLOCK_M)
    row_mask = rows < num_valid_rows
    packed = tl.load(sorted_token_ids_ptr + rows, mask=row_mask, other=0)
    token = packed & 0x00FFFFFF
    slot = (packed.to(tl.uint32) >> 24).to(tl.int32)
    route_mask = row_mask & (token < token_num) & (slot < TOPK)
    route = token.to(tl.int64) * TOPK + slot.to(tl.int64)

    expert = tl.load(sorted_expert_ids_ptr + pid_m).to(tl.int64)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < I
    k_offsets = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_base in range(0, D, BLOCK_K):
        k = k_base + k_offsets
        k_mask = k < D
        a_ptrs = dout_ptr + token[:, None].to(tl.int64) * D + k[None, :]
        b_ptrs = (
            w2_ptr
            + expert * D * I
            + k[:, None].to(tl.int64) * I
            + cols[None, :]
        )
        a = tl.load(
            a_ptrs,
            mask=route_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(a, b, acc=acc)

    z_base = route[:, None] * (2 * I) + cols[None, :]
    z_mask = route_mask[:, None] & col_mask[None, :]
    z_gate = tl.load(z_ptr + z_base, mask=z_mask, other=0.0).to(tl.float32)
    z_up = tl.load(z_ptr + z_base + I, mask=z_mask, other=0.0).to(tl.float32)
    score = tl.load(score_ptr + route, mask=route_mask, other=0.0).to(tl.float32)

    sigmoid = tl.sigmoid(z_gate)
    silu = z_gate * sigmoid
    activation = tl.where(col_mask[None, :], silu * z_up, 0.0)
    q = acc * score[:, None]
    dz_gate = q * z_up * sigmoid * (1.0 + z_gate * (1.0 - sigmoid))
    dz_up = q * silu

    tl.store(dz_ptr + z_base, dz_gate, mask=z_mask)
    tl.store(dz_ptr + z_base + I, dz_up, mask=z_mask)
    tl.store(
        a_scaled_ptr + route[:, None] * I + cols[None, :],
        score[:, None] * activation,
        mask=z_mask,
    )

    ds_partial = tl.sum(acc * activation, axis=1)
    tl.store(
        ds_partial_ptr + route * DS_PARTS + pid_n,
        ds_partial,
        mask=route_mask,
    )


@triton.jit
def _moe_ds_reduce_kernel(
    ds_partial_ptr,
    ds_ptr,
    route_num,
    DS_PARTS: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    routes = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    mask = routes < route_num
    acc = tl.zeros((BLOCK_R,), dtype=tl.float32)
    for part in range(DS_PARTS):
        acc += tl.load(
            ds_partial_ptr + routes * DS_PARTS + part,
            mask=mask,
            other=0.0,
        )
    tl.store(ds_ptr + routes, acc, mask=mask)


@triton.jit
def _moe_route_dx_kernel(
    dz_ptr,
    w1_ptr,
    sorted_token_ids_ptr,
    sorted_expert_ids_ptr,
    num_valid_ids_ptr,
    dx_route_ptr,
    token_num,
    D: tl.constexpr,
    I2: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_base = pid_m * BLOCK_M
    num_valid_rows = tl.load(num_valid_ids_ptr)
    if row_base >= num_valid_rows:
        return

    rows = row_base + tl.arange(0, BLOCK_M)
    row_mask = rows < num_valid_rows
    packed = tl.load(sorted_token_ids_ptr + rows, mask=row_mask, other=0)
    token = packed & 0x00FFFFFF
    slot = (packed.to(tl.uint32) >> 24).to(tl.int32)
    route_mask = row_mask & (token < token_num) & (slot < TOPK)
    route = token.to(tl.int64) * TOPK + slot.to(tl.int64)
    expert = tl.load(sorted_expert_ids_ptr + pid_m).to(tl.int64)

    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < D
    k_offsets = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_base in range(0, I2, BLOCK_K):
        k = k_base + k_offsets
        k_mask = k < I2
        a_ptrs = dz_ptr + route[:, None] * I2 + k[None, :]
        b_ptrs = (
            w1_ptr
            + expert * I2 * D
            + k[:, None].to(tl.int64) * D
            + cols[None, :]
        )
        a = tl.load(
            a_ptrs,
            mask=route_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(a, b, acc=acc)

    tl.store(
        dx_route_ptr + route[:, None] * D + cols[None, :],
        acc,
        mask=route_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def _moe_route_reduce_kernel(
    dx_route_ptr,
    dx_ptr,
    token_num,
    D: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (tokens[:, None] < token_num) & (cols[None, :] < D)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for slot in range(TOPK):
        route = tokens * TOPK + slot
        acc += tl.load(
            dx_route_ptr + route[:, None].to(tl.int64) * D + cols[None, :],
            mask=mask,
            other=0.0,
        )
    tl.store(
        dx_ptr + tokens[:, None].to(tl.int64) * D + cols[None, :],
        acc,
        mask=mask,
    )


@triton.jit
def _moe_dw1_kernel(
    x_ptr,
    dz_ptr,
    compact_route_ids_ptr,
    expert_offsets_ptr,
    dw1_ptr,
    D: tl.constexpr,
    I2: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < I2
    col_mask = cols < D

    start = tl.load(expert_offsets_ptr + expert)
    end = tl.load(expert_offsets_ptr + expert + 1)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    route_pos = start
    k_offsets = tl.arange(0, BLOCK_K)
    while route_pos < end:
        positions = route_pos + k_offsets
        k_mask = positions < end
        routes = tl.load(compact_route_ids_ptr + positions, mask=k_mask, other=0)
        tokens = routes // TOPK
        a = tl.load(
            dz_ptr + routes[None, :].to(tl.int64) * I2 + rows[:, None],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            x_ptr + tokens[:, None].to(tl.int64) * D + cols[None, :],
            mask=k_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(a, b, acc=acc)
        route_pos += BLOCK_K

    tl.store(
        dw1_ptr
        + expert * I2 * D
        + rows[:, None].to(tl.int64) * D
        + cols[None, :],
        acc,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def _moe_dw2_kernel(
    dout_ptr,
    a_scaled_ptr,
    compact_route_ids_ptr,
    expert_offsets_ptr,
    dw2_ptr,
    D: tl.constexpr,
    I: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < D
    col_mask = cols < I

    start = tl.load(expert_offsets_ptr + expert)
    end = tl.load(expert_offsets_ptr + expert + 1)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    route_pos = start
    k_offsets = tl.arange(0, BLOCK_K)
    while route_pos < end:
        positions = route_pos + k_offsets
        k_mask = positions < end
        routes = tl.load(compact_route_ids_ptr + positions, mask=k_mask, other=0)
        tokens = routes // TOPK
        a = tl.load(
            dout_ptr + tokens[None, :].to(tl.int64) * D + rows[:, None],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            a_scaled_ptr + routes[:, None].to(tl.int64) * I + cols[None, :],
            mask=k_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(a, b, acc=acc)
        route_pos += BLOCK_K

    tl.store(
        dw2_ptr
        + expert * D * I
        + rows[:, None].to(tl.int64) * I
        + cols[None, :],
        acc,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@dataclass(frozen=True)
class TritonMoeBackwardConfig:
    """Compile-time tile choices for the prototype kernels."""

    route_block_m: int = 32
    down_block_n: int = 128
    down_block_k: int = 64
    dx_block_n: int = 128
    dx_block_k: int = 64
    reduce_block_m: int = 16
    reduce_block_n: int = 128
    dw1_block_m: int = 64
    dw1_block_n: int = 128
    dw1_block_k: int = 32
    dw2_block_m: int = 64
    dw2_block_n: int = 64
    dw2_block_k: int = 64
    num_warps: int = 4
    num_stages: int = 2
    waves_per_eu: int = 0


DEFAULT_CONFIG = TritonMoeBackwardConfig()


@dataclass
class TritonMoeBackwardMetadata:
    sorted_token_ids: torch.Tensor
    sorted_weights: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor
    reverse_sorted: torch.Tensor
    compact_route_ids: torch.Tensor
    expert_offsets: torch.Tensor
    expert_padded_offsets: torch.Tensor
    block_m: int


@dataclass
class TritonMoeBackwardWorkspace:
    d_z: torch.Tensor
    a_scaled: torch.Tensor
    d_s_partial: torch.Tensor
    d_s: torch.Tensor
    d_x_route: torch.Tensor
    d_x: torch.Tensor
    d_w1: torch.Tensor
    d_w2: torch.Tensor


def _check_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tensor.stride()}")


def prepare_moe_backward_metadata(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    model_dim: int,
    *,
    block_m: int = DEFAULT_CONFIG.route_block_m,
) -> TritonMoeBackwardMetadata:
    """Build reusable forward metadata.

    The Opus sorter emits padded token/slot IDs and reverse_sorted. The compact
    route list is the exact expert-major order recovered from reverse_sorted;
    it is a temporary prototype ABI for varlen-K dW1/dW2.
    """

    if topk_ids.dtype != torch.int32:
        raise TypeError(f"topk_ids must be int32, got {topk_ids.dtype}")
    if topk_weights.dtype != torch.float32:
        raise TypeError(f"topk_weights must be float32, got {topk_weights.dtype}")
    if topk_ids.shape != topk_weights.shape or topk_ids.ndim != 2:
        raise ValueError("topk_ids and topk_weights must have the same [T,K] shape")
    _check_contiguous("topk_ids", topk_ids)
    _check_contiguous("topk_weights", topk_weights)

    # Call the standard Opus sorter at its low-level API. The public dispatcher
    # may route output_aux=True through the MXFP4 adaptive sorter, whose codegen
    # shape set is unrelated to this BF16 backward prototype.
    import aiter

    token_num, topk = topk_ids.shape
    max_num_tokens_padded = int(topk_ids.numel() + num_experts * block_m - topk)
    max_num_m_blocks = (max_num_tokens_padded + block_m - 1) // block_m
    sorted_token_ids = torch.empty(
        max_num_tokens_padded, dtype=torch.int32, device=topk_ids.device
    )
    sorted_weights = torch.empty(
        max_num_tokens_padded, dtype=torch.float32, device=topk_ids.device
    )
    sorted_expert_ids = torch.empty(
        max_num_m_blocks, dtype=torch.int32, device=topk_ids.device
    )
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=topk_ids.device)
    moe_buf = torch.empty((0, 0), dtype=torch.bfloat16, device=topk_ids.device)
    m_indices = torch.empty(
        max_num_tokens_padded, dtype=torch.int32, device=topk_ids.device
    )
    reverse_sorted = torch.empty(
        token_num * topk, dtype=torch.int32, device=topk_ids.device
    )
    workspace_size = aiter.moe_sorting_opus_get_workspace_size(
        token_num, num_experts, topk, 1
    )
    sorting_workspace = (
        torch.empty(workspace_size, dtype=torch.uint8, device=topk_ids.device)
        if workspace_size > 0
        else None
    )
    aiter.moe_sorting_opus_fwd(
        topk_ids,
        topk_weights,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        num_experts,
        block_m,
        None,
        None,
        sorting_workspace,
        1,
        None,
        m_indices,
        reverse_sorted,
    )

    compact_route_ids = torch.argsort(reverse_sorted).to(torch.int32)
    counts = torch.bincount(
        topk_ids.reshape(-1).to(torch.int64), minlength=num_experts
    )
    expert_offsets = torch.empty(
        (num_experts + 1,), dtype=torch.int32, device=topk_ids.device
    )
    expert_offsets[0] = 0
    expert_offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
    padded_counts = torch.div(
        counts + block_m - 1, block_m, rounding_mode="floor"
    ) * block_m
    expert_padded_offsets = torch.empty_like(expert_offsets)
    expert_padded_offsets[0] = 0
    expert_padded_offsets[1:] = torch.cumsum(padded_counts, dim=0).to(torch.int32)

    return TritonMoeBackwardMetadata(
        sorted_token_ids=sorted_token_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        reverse_sorted=reverse_sorted,
        compact_route_ids=compact_route_ids,
        expert_offsets=expert_offsets,
        expert_padded_offsets=expert_padded_offsets,
        block_m=block_m,
    )


def allocate_moe_backward_workspace(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    config: TritonMoeBackwardConfig = DEFAULT_CONFIG,
) -> TritonMoeBackwardWorkspace:
    token_num, model_dim = x.shape
    num_experts, gate_up_dim, w1_model_dim = w1.shape
    w2_experts, w2_model_dim, inter_dim = w2.shape
    if w1_model_dim != model_dim or w2_model_dim != model_dim:
        raise ValueError("W1/W2 model dimensions must match X")
    if w2_experts != num_experts or gate_up_dim != 2 * inter_dim:
        raise ValueError("expected W1=[E,2I,D] and W2=[E,D,I]")
    if topk_weights.shape[0] != token_num:
        raise ValueError("topk_weights token dimension must match X")
    topk = topk_weights.shape[1]
    route_num = token_num * topk
    ds_parts = triton.cdiv(inter_dim, config.down_block_n)
    kwargs = {"device": x.device, "dtype": x.dtype}
    return TritonMoeBackwardWorkspace(
        d_z=torch.empty((route_num, gate_up_dim), **kwargs),
        a_scaled=torch.empty((route_num, inter_dim), **kwargs),
        d_s_partial=torch.empty(
            (route_num, ds_parts), device=x.device, dtype=torch.float32
        ),
        d_s=torch.empty_like(topk_weights, dtype=torch.float32),
        d_x_route=torch.empty((route_num, model_dim), **kwargs),
        d_x=torch.empty_like(x),
        d_w1=torch.empty_like(w1),
        d_w2=torch.empty_like(w2),
    )


def _validate_inputs(
    dout: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    metadata: TritonMoeBackwardMetadata,
    workspace: TritonMoeBackwardWorkspace,
    config: TritonMoeBackwardConfig,
) -> tuple[int, int, int, int, int]:
    if x.dtype != torch.bfloat16 or any(
        t.dtype != torch.bfloat16 for t in (dout, z, w1, w2)
    ):
        raise TypeError("the first Triton prototype supports BF16 tensors only")
    for name, tensor in (
        ("dout", dout),
        ("x", x),
        ("z", z),
        ("w1", w1),
        ("w2", w2),
        ("topk_weights", topk_weights),
    ):
        _check_contiguous(name, tensor)
    token_num, model_dim = x.shape
    num_experts, gate_up_dim, _ = w1.shape
    _, _, inter_dim = w2.shape
    topk = topk_weights.shape[1]
    route_num = token_num * topk
    if dout.shape != x.shape:
        raise ValueError("dout must have shape [T,D]")
    if w1.shape != (num_experts, 2 * inter_dim, model_dim):
        raise ValueError("W1 must have shape [E,2I,D]")
    if w2.shape != (num_experts, model_dim, inter_dim):
        raise ValueError("W2 must have shape [E,D,I]")
    if z.shape != (route_num, 2 * inter_dim):
        raise ValueError("Z must have token-slot-major shape [T*K,2I]")
    if metadata.block_m != config.route_block_m:
        raise ValueError("sorting block_m must match Triton route_block_m")
    if workspace.d_z.shape != z.shape:
        raise ValueError("workspace was allocated for a different shape")
    return num_experts, token_num, topk, model_dim, inter_dim


def launch_down_backward(
    dout: torch.Tensor,
    z: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    metadata: TritonMoeBackwardMetadata,
    workspace: TritonMoeBackwardWorkspace,
    *,
    config: TritonMoeBackwardConfig = DEFAULT_CONFIG,
) -> None:
    token_num, model_dim = dout.shape
    inter_dim = w2.shape[2]
    topk = topk_weights.shape[1]
    route_num = token_num * topk
    ds_parts = triton.cdiv(inter_dim, config.down_block_n)
    grid = (
        triton.cdiv(metadata.sorted_token_ids.numel(), config.route_block_m),
        ds_parts,
    )
    _moe_down_bwd_fused_kernel[grid](
        dout,
        z,
        w2,
        topk_weights,
        metadata.sorted_token_ids,
        metadata.sorted_expert_ids,
        metadata.num_valid_ids,
        workspace.d_z,
        workspace.a_scaled,
        workspace.d_s_partial,
        token_num,
        D=model_dim,
        I=inter_dim,
        TOPK=topk,
        DS_PARTS=ds_parts,
        BLOCK_M=config.route_block_m,
        BLOCK_N=config.down_block_n,
        BLOCK_K=config.down_block_k,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        waves_per_eu=config.waves_per_eu,
    )
    _moe_ds_reduce_kernel[(triton.cdiv(route_num, 256),)](
        workspace.d_s_partial,
        workspace.d_s,
        route_num,
        DS_PARTS=ds_parts,
        BLOCK_R=256,
        num_warps=4,
    )


def launch_route_dx(
    x: torch.Tensor,
    w1: torch.Tensor,
    topk_weights: torch.Tensor,
    metadata: TritonMoeBackwardMetadata,
    workspace: TritonMoeBackwardWorkspace,
    *,
    config: TritonMoeBackwardConfig = DEFAULT_CONFIG,
) -> None:
    token_num, model_dim = x.shape
    topk = topk_weights.shape[1]
    gate_up_dim = w1.shape[1]
    route_grid = (
        triton.cdiv(metadata.sorted_token_ids.numel(), config.route_block_m),
        triton.cdiv(model_dim, config.dx_block_n),
    )
    _moe_route_dx_kernel[route_grid](
        workspace.d_z,
        w1,
        metadata.sorted_token_ids,
        metadata.sorted_expert_ids,
        metadata.num_valid_ids,
        workspace.d_x_route,
        token_num,
        D=model_dim,
        I2=gate_up_dim,
        TOPK=topk,
        BLOCK_M=config.route_block_m,
        BLOCK_N=config.dx_block_n,
        BLOCK_K=config.dx_block_k,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        waves_per_eu=config.waves_per_eu,
    )
    reduce_grid = (
        triton.cdiv(token_num, config.reduce_block_m),
        triton.cdiv(model_dim, config.reduce_block_n),
    )
    _moe_route_reduce_kernel[reduce_grid](
        workspace.d_x_route,
        workspace.d_x,
        token_num,
        D=model_dim,
        TOPK=topk,
        BLOCK_M=config.reduce_block_m,
        BLOCK_N=config.reduce_block_n,
        num_warps=4,
    )


def launch_dw1(
    x: torch.Tensor,
    w1: torch.Tensor,
    topk_weights: torch.Tensor,
    metadata: TritonMoeBackwardMetadata,
    workspace: TritonMoeBackwardWorkspace,
    *,
    config: TritonMoeBackwardConfig = DEFAULT_CONFIG,
) -> None:
    num_experts, gate_up_dim, model_dim = w1.shape
    topk = topk_weights.shape[1]
    grid = (
        num_experts,
        triton.cdiv(gate_up_dim, config.dw1_block_m),
        triton.cdiv(model_dim, config.dw1_block_n),
    )
    _moe_dw1_kernel[grid](
        x,
        workspace.d_z,
        metadata.compact_route_ids,
        metadata.expert_offsets,
        workspace.d_w1,
        D=model_dim,
        I2=gate_up_dim,
        TOPK=topk,
        BLOCK_M=config.dw1_block_m,
        BLOCK_N=config.dw1_block_n,
        BLOCK_K=config.dw1_block_k,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        waves_per_eu=config.waves_per_eu,
    )


def launch_dw2(
    dout: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    metadata: TritonMoeBackwardMetadata,
    workspace: TritonMoeBackwardWorkspace,
    *,
    config: TritonMoeBackwardConfig = DEFAULT_CONFIG,
) -> None:
    num_experts, model_dim, inter_dim = w2.shape
    topk = topk_weights.shape[1]
    grid = (
        num_experts,
        triton.cdiv(model_dim, config.dw2_block_m),
        triton.cdiv(inter_dim, config.dw2_block_n),
    )
    _moe_dw2_kernel[grid](
        dout,
        workspace.a_scaled,
        metadata.compact_route_ids,
        metadata.expert_offsets,
        workspace.d_w2,
        D=model_dim,
        I=inter_dim,
        TOPK=topk,
        BLOCK_M=config.dw2_block_m,
        BLOCK_N=config.dw2_block_n,
        BLOCK_K=config.dw2_block_k,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        waves_per_eu=config.waves_per_eu,
    )


def triton_moe_backward_out(
    dout: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    metadata: TritonMoeBackwardMetadata,
    workspace: TritonMoeBackwardWorkspace,
    *,
    config: TritonMoeBackwardConfig = DEFAULT_CONFIG,
) -> TritonMoeBackwardWorkspace:
    """Run K1-K5 into preallocated output/workspace tensors."""

    _validate_inputs(
        dout, x, z, w1, w2, topk_weights, metadata, workspace, config
    )
    launch_down_backward(
        dout, z, w2, topk_weights, metadata, workspace, config=config
    )
    launch_route_dx(x, w1, topk_weights, metadata, workspace, config=config)
    launch_dw1(x, w1, topk_weights, metadata, workspace, config=config)
    launch_dw2(dout, w2, topk_weights, metadata, workspace, config=config)
    return workspace


def triton_moe_backward(
    dout: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    metadata: TritonMoeBackwardMetadata,
    *,
    config: TritonMoeBackwardConfig = DEFAULT_CONFIG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocating convenience wrapper returning dX, dW1, dW2 and dS."""

    workspace = allocate_moe_backward_workspace(
        x, w1, w2, topk_weights, config=config
    )
    triton_moe_backward_out(
        dout,
        x,
        z,
        w1,
        w2,
        topk_weights,
        metadata,
        workspace,
        config=config,
    )
    return workspace.d_x, workspace.d_w1, workspace.d_w2, workspace.d_s


__all__ = [
    "DEFAULT_CONFIG",
    "TritonMoeBackwardConfig",
    "TritonMoeBackwardMetadata",
    "TritonMoeBackwardWorkspace",
    "allocate_moe_backward_workspace",
    "launch_down_backward",
    "launch_dw1",
    "launch_dw2",
    "launch_route_dx",
    "prepare_moe_backward_metadata",
    "triton_moe_backward",
    "triton_moe_backward_out",
]
