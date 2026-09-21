# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.sonicmoe_config_utils import (
    get_sonicmoe_kernel_config,
    split_launch_config,
)

from .activation_kernels import activation_bwd, activation_fwd
from .enums import LIBRARY_NAME
from .grouped_gemm_triton import grouped_gemm
from .reduction_over_k_gather import token_gather_and_sum_varlen_K_triton

_db2_and_ds_repr = make_kernel_repr(
    "sonicmoe_db2_and_ds",
    ["H", "E", "OLD_DS_PARTIAL_N", "BLOCK_H", "BLOCK_TK", "BLOCK_OLD_DS_PARTIAL_N"],
)
_db1_repr = make_kernel_repr(
    "sonicmoe_db1", ["I", "E", "BLOCK_I", "BLOCK_TK", "CONCAT_LAYOUT"]
)


@triton.jit(repr=_db2_and_ds_repr)
def db2_and_ds_kernel(
    dout_ptr,
    s_ptr,
    new_ds_partial_ptr,
    old_ds_partial_ptr,
    b2_ptr,
    db2_ptr,
    x_gather_idx_ptr,
    s_scatter_idx_ptr,
    expert_offset_ptr,
    H: tl.constexpr,
    E: tl.constexpr,
    OLD_DS_PARTIAL_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_TK: tl.constexpr,
    BLOCK_OLD_DS_PARTIAL_N: tl.constexpr,
):
    Eidx = tl.program_id(0)
    Hidx = tl.program_id(1)
    NUM_H_BLOCKS: tl.constexpr = tl.num_programs(1)

    h_offsets = Hidx * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    E_count_start = tl.load(expert_offset_ptr + Eidx)
    E_count_end = tl.load(expert_offset_ptr + Eidx + 1)
    n_tokens = E_count_end - E_count_start

    b2 = tl.load(b2_ptr + Eidx * H + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
    db2_acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for block_start in tl.range(0, n_tokens, BLOCK_TK):
        tk_offsets = block_start + tl.arange(0, BLOCK_TK)
        tk_mask = tk_offsets < n_tokens
        tk_grouped = E_count_start + tk_offsets

        token_indices = tl.load(
            x_gather_idx_ptr + tk_grouped, mask=tk_mask, other=0
        ).to(tl.int64)
        scatter_indices = tl.load(
            s_scatter_idx_ptr + tk_grouped, mask=tk_mask, other=0
        ).to(tl.int64)
        s = tl.load(s_ptr + scatter_indices, mask=tk_mask, other=0.0).to(tl.float32)

        dout_offsets = token_indices[:, None] * H + h_offsets[None, :]
        dout_mask = tk_mask[:, None] & h_mask[None, :]
        dout = tl.load(dout_ptr + dout_offsets, mask=dout_mask, other=0.0).to(
            tl.float32
        )

        db2_acc += tl.sum(dout * s[:, None], axis=0)

        ds_partial = tl.sum(dout * b2[None, :], axis=1)

        if Hidx == 0:
            n_offsets = tl.arange(0, BLOCK_OLD_DS_PARTIAL_N)
            old_ds_partial_offsets = (
                scatter_indices[:, None] * OLD_DS_PARTIAL_N + n_offsets[None, :]
            )
            old_ds_partial_mask = tk_mask[:, None] & (
                n_offsets[None, :] < OLD_DS_PARTIAL_N
            )
            old_ds_partial_vals = tl.load(
                old_ds_partial_ptr + old_ds_partial_offsets,
                mask=old_ds_partial_mask,
                other=0.0,
            ).to(tl.float32)
            ds_partial += tl.sum(old_ds_partial_vals, axis=1)

        tl.store(
            new_ds_partial_ptr + scatter_indices * NUM_H_BLOCKS + Hidx,
            ds_partial,
            mask=tk_mask,
        )

    tl.store(db2_ptr + Eidx * H + h_offsets, db2_acc, mask=h_mask)


@triton.jit(repr=_db1_repr)
def db1_kernel(
    dh_ptr,
    db1_ptr,
    expert_offset_ptr,
    I: tl.constexpr,
    E: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_TK: tl.constexpr,
    CONCAT_LAYOUT: tl.constexpr = False,
):
    Eidx = tl.program_id(0)

    E_count_start = tl.load(expert_offset_ptr + Eidx).to(tl.int64)
    E_count_end = tl.load(expert_offset_ptr + Eidx + 1).to(tl.int64)
    n_tokens = E_count_end - E_count_start

    NUM_I_BLOCKS: tl.constexpr = triton.cdiv(I, BLOCK_I)
    I_HALF: tl.constexpr = I // 2
    for Iidx in tl.static_range(0, NUM_I_BLOCKS, 1):
        i_offsets = Iidx * BLOCK_I + tl.arange(0, BLOCK_I)
        i_mask = i_offsets < I

        db1_acc = tl.zeros([BLOCK_I], dtype=tl.float32)

        for block_start in tl.range(0, n_tokens, BLOCK_TK):
            tk_offsets = block_start + tl.arange(0, BLOCK_TK)
            tk_mask = tk_offsets < n_tokens
            tk_grouped = E_count_start + tk_offsets

            dz_offsets = tk_grouped[:, None] * I + i_offsets[None, :]
            dz_mask = tk_mask[:, None] & i_mask[None, :]
            dz = tl.load(dh_ptr + dz_offsets, mask=dz_mask, other=0.0).to(tl.float32)
            db1_acc += tl.sum(dz, axis=0)

        if CONCAT_LAYOUT:
            out_offsets = i_offsets // 2 + (i_offsets % 2) * I_HALF
        else:
            out_offsets = i_offsets
        db1_offsets = Eidx.to(tl.int64) * I + out_offsets
        tl.store(db1_ptr + db1_offsets, db1_acc, mask=i_mask)


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_up_projection_backward_act_rocm",
    mutates_args={"dx_expanded", "db1"},
)
def _up_projection_backward_act(
    w1: torch.Tensor,
    dx_expanded: torch.Tensor,
    dh: torch.Tensor,
    db1: torch.Tensor | None,
    expert_frequency_offset: torch.Tensor,
    is_glu_activation: bool,
    concat_layout: bool = False,
    grouped_weight_layout: bool = False,
) -> None:
    if grouped_weight_layout:
        E, _, I_full = w1.size()
        gemm_w1 = w1
    else:
        I_full, _, E = w1.size()
        gemm_w1 = w1.permute(2, 0, 1)
    I = I_full // 2 if is_glu_activation else I_full

    grouped_gemm(
        dh,
        gemm_w1,
        expert_frequency_offset,
        out=dx_expanded,
        B_is_transposed=grouped_weight_layout,
    )

    if db1 is not None:
        db1_cfg = get_sonicmoe_kernel_config("db1_kernel")
        constexprs, launch = split_launch_config(db1_cfg)
        db1_kernel[(E,)](
            dh,
            db1,
            expert_frequency_offset,
            (2 * I if is_glu_activation else I),
            E,
            CONCAT_LAYOUT=concat_layout and is_glu_activation,
            **constexprs,
            **launch,
        )


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_down_projection_backward_act_rocm",
    mutates_args={"dh", "ds", "db2", "a_prime"},
)
def _down_projection_backward_act(
    dout: torch.Tensor,
    h: torch.Tensor,
    w2: torch.Tensor,
    dh: torch.Tensor,
    ds: torch.Tensor,
    b2: torch.Tensor | None,
    db2: torch.Tensor | None,
    a_prime: torch.Tensor,
    topk_scores: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    s_scatter_idx: torch.Tensor,
    activation_type: str,
    grouped_weight_layout: bool = False,
    concat_layout: bool = False,
) -> None:
    if grouped_weight_layout:
        E, I, H = w2.size()
        gemm_w2 = w2
    else:
        H, I, E = w2.size()
        gemm_w2 = w2.permute(2, 0, 1)
    TK = x_gather_idx.size(0)
    s = topk_scores[s_scatter_idx]

    # Compute u = dout @ w2.T once. The router gradient reuses this GEMM:
    # dot(dout, a @ w2) == dot(a, dout @ w2.T), while da = score * u.
    dout_gathered = dout[x_gather_idx]
    dh_unscaled = torch.empty(TK, I, dtype=dh.dtype, device=dh.device)
    grouped_gemm(
        dout_gathered,
        gemm_w2,
        expert_frequency_offset,
        out=dh_unscaled,
        B_is_transposed=grouped_weight_layout,
    )

    a_prime_val = activation_fwd(h, I, activation_type, concat_layout)
    a_prime.copy_(a_prime_val)
    ds_scattered = (a_prime_val.float() * dh_unscaled.float()).sum(dim=-1)

    dh_raw = dh_unscaled * s.unsqueeze(-1)
    dh_act = activation_bwd(h, dh_raw, I, activation_type, concat_layout)
    dh.copy_(dh_act)

    if db2 is None:
        ds[s_scatter_idx] = ds_scattered
    else:
        old_ds_partial = torch.empty(
            TK, 1, device=ds_scattered.device, dtype=ds_scattered.dtype
        )
        old_ds_partial[s_scatter_idx, 0] = ds_scattered

        db2_cfg = get_sonicmoe_kernel_config("db2_and_ds_kernel")
        block_h_max = db2_cfg.pop("BLOCK_H_MAX")
        BLOCK_H = min(triton.next_power_of_2(H), block_h_max)
        NUM_H_BLOCKS = triton.cdiv(H, BLOCK_H)
        new_ds_partial = torch.empty(
            TK, NUM_H_BLOCKS, dtype=torch.float32, device=ds.device
        )

        constexprs, launch = split_launch_config(db2_cfg)
        db2_and_ds_kernel[(E, NUM_H_BLOCKS)](
            dout,
            topk_scores,
            new_ds_partial,
            old_ds_partial,
            b2,
            db2,
            x_gather_idx,
            s_scatter_idx,
            expert_frequency_offset,
            H,
            E,
            1,
            BLOCK_H=BLOCK_H,
            **constexprs,
            **launch,
        )

        if NUM_H_BLOCKS == 1:
            ds.copy_(new_ds_partial.view(-1).to(dtype=ds.dtype))
        else:
            ds.copy_(new_ds_partial.sum(dim=-1, dtype=ds.dtype))


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_token_broadcast_backward_rocm", mutates_args={"dx_reduced"}
)
def _token_broadcast_backward(
    dx_reduced: torch.Tensor,
    dx_expanded: torch.Tensor,
    s_reverse_scatter_idx: torch.Tensor,
    num_activated_expert_per_token_offset: torch.Tensor | None,
    varlen_K_max: int,
    H: int,
    is_varlen_K: bool,
) -> None:
    if num_activated_expert_per_token_offset is None:
        assert not is_varlen_K
    token_gather_and_sum_varlen_K_triton(
        dx_expanded,
        None,
        dx_reduced,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        dx_reduced.size(0),
        varlen_K_max,
        H,
        is_varlen_K,
    )
