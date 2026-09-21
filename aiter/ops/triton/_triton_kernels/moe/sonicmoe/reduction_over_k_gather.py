# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************


import torch
import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.sonicmoe_config_utils import (
    get_token_gather_config,
    split_launch_config,
)

_token_gather_sum_repr = make_kernel_repr(
    "sonicmoe_token_gather_sum",
    [
        "H",
        "MAX_K",
        "BLOCK_H",
        "BLOCK_K",
        "w_is_None",
        "is_varlen_K",
    ],
)


@triton.jit(repr=_token_gather_sum_repr)
def token_gather_sum_kernel(
    x_ptr,  # (Mtotal, H)
    w_ptr,  # (Mtotal,)
    M_perm_ptr,  # (Mtotal,) int32
    M_offset_ptr,  # (T+1,)   int32
    out_ptr,  # (T, H)
    T,
    H: tl.constexpr,
    MAX_K: tl.constexpr,
    stride_xM: tl.constexpr,
    stride_xH: tl.constexpr,
    stride_outT: tl.constexpr,
    stride_outH: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    w_is_None: tl.constexpr,
    is_varlen_K: tl.constexpr,
):
    pid_t = tl.program_id(axis=0)
    t_idx = pid_t.to(tl.int64)

    if is_varlen_K:
        Ms = tl.load(M_offset_ptr + t_idx).to(tl.int64)
        Me = tl.load(M_offset_ptr + t_idx + 1).to(tl.int64)
        K_this_token = Me - Ms  # actual K for this token
    else:
        Ms = MAX_K * t_idx
        K_this_token: tl.constexpr = MAX_K

    for h_tile in tl.static_range(triton.cdiv(H, BLOCK_H)):
        h_idx = (h_tile * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.int64)  # [BLOCK_H]
        m_h = h_idx < H

        acc = tl.zeros([BLOCK_H], dtype=tl.float32)  # [BLOCK_H]

        for k_tile in tl.range(tl.cdiv(K_this_token, BLOCK_K)):
            k_offset = k_tile * BLOCK_K

            k_idx = (k_offset + tl.arange(0, BLOCK_K)).to(tl.int64)  # [BLOCK_K]

            m_k = k_idx < K_this_token  # [BLOCK_K]

            m_abs = Ms + k_idx  # [BLOCK_K]

            perm_idx = tl.load(M_perm_ptr + m_abs, mask=m_k, other=0).to(
                tl.int64
            )  # [BLOCK_K]

            x_ptrs = x_ptr + perm_idx[:, None] * stride_xM + h_idx[None, :] * stride_xH
            x_mask = m_k[:, None] & m_h[None, :]
            x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

            if w_is_None:
                acc += tl.sum(x_vals, axis=0)  # [BLOCK_H]
            else:
                w_vals = tl.load(w_ptr + m_abs, mask=m_k, other=0.0).to(
                    tl.float32
                )  # [BLOCK_K]
                acc += tl.sum(x_vals * w_vals[:, None], axis=0)  # [BLOCK_H]

        out_ptrs = out_ptr + t_idx * stride_outT + h_idx * stride_outH
        tl.store(out_ptrs, acc, mask=m_h)


def token_gather_and_sum_varlen_K_triton(
    x: torch.Tensor,  # (Mtotal, H)
    w: torch.Tensor | None,  # (Mtotal,)
    out: torch.Tensor,  # (T, H)
    M_perm: torch.Tensor,  # (Mtotal,) int32
    M_offset: torch.Tensor,  # (T+1,)   int32, variable K per token
    T: int,
    MAX_K: int,  # maximum K across all tokens
    H: int,
    is_varlen_K: bool,
):
    """Gather and reduce a variable number of weighted rows per token."""
    common = (x, w, M_perm, M_offset, out)
    kwargs = {
        "T": T,
        "H": H,
        "MAX_K": MAX_K,
        "stride_xM": x.stride(0),
        "stride_xH": x.stride(1),
        "stride_outT": out.stride(0),
        "stride_outH": out.stride(1),
        "w_is_None": (w is None),
        "is_varlen_K": is_varlen_K,
    }
    constexprs, launch = split_launch_config(get_token_gather_config(H))
    token_gather_sum_kernel[(T,)](*common, **kwargs, **constexprs, **launch)
