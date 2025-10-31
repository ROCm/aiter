# Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
# Copyright (C) 2024-2025, The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
import functools
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.common_utils import (
    prev_power_of_2,
    autotune_max_seq_len,
    switch_to_contiguous_if_needed,
)
from aiter.ops.triton._triton_kernels.hstu_attention import (
    _hstu_attn_fwd,
    _get_fwd_config,
    _hstu_attn_bwd,
    _get_bwd_config,
)


try:
    from triton.language.extra.libdevice import (
        fast_dividef,
        fast_expf,
    )  # @manual=//triton:triton
except ImportError:
    try:
        # @manual=//triton:triton
        from triton.language.extra.hip.libdevice import fast_dividef, fast_expf
    except ImportError:
        # pyre-ignore[21]
        from triton.language.math import (
            fast_dividef,
            fast_expf,
        )  # @manual=//triton:triton
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def triton_hstu_attention_fwd(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool,
    num_targets: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    sort_by_length_indices: Optional[torch.Tensor],
    config: Optional[dict] = None,
) -> torch.Tensor:
    """
    Computes HSTU attention fwd pass, compute the math dot(silu(dot(q * trans(k))) * v). inputs q, kv are of the jagged formats

    Key parameters:
    - N: max sequence length
    - alpha: scale parameter to multiply output of first dot
    - q: tensor with shape (L, H, D), L are sum of lengths of all sequences
    - k: tensor with shape (L, H, D), L are sum of lengths of all sequences
    - v: tensor with shape (L, H, D), L are sum of lengths of all sequences
    - seq_offsets: tensor with shape (B + 1), indicates lengths of each sequences.
    - causal: whether use causal mask.
    - num_targets: number of targets.
    - contextual_seq_len: contexual sequence length.
    - sort_by_length_indices: indices of sequences sorted by lengths
    - config: Optional, tuning configs to run the kernel

    Returns:
    - Y: output with the shape (L, H, D).
    """
    _LOGGER.info(
        f"HSTU_ATTENTION_FWD: N={N} alpha={alpha} q={tuple(q.shape)} k={tuple(k.shape)}  v={tuple(v.shape)} seq_offsets={tuple(seq_offsets.shape)}"
    )
    Z = seq_offsets.numel() - 1
    AUTOTUNE_Z = prev_power_of_2(Z)
    L, H, DimQ = q.shape
    _, _, DimV = v.shape
    out = torch.empty_like(v)
    has_multiple_targets = num_targets is not None
    has_contextual_seq_len = contextual_seq_len > 0
    has_max_attn_len = max_attn_len > 0
    has_sort_by_length_indices = sort_by_length_indices is not None
    if L == 0:
        return out

    max_seq_len = autotune_max_seq_len(N)
    DeltaSize = 0
    IS_DELTA_Q = False

    if config is None:
        config = _get_fwd_config(
            AUTOTUNE_Z, H, max_seq_len, DimQ, DimV, DeltaSize, IS_DELTA_Q
        )

    grid = lambda meta: (  # noqa E731
        triton.cdiv(N, meta["BLOCK_M"]),
        Z * H,
    )

    _hstu_attn_fwd[grid](
        Q=q,
        K=k,
        V=v,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        Out=out,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_om=out.stride(0),
        stride_oh=out.stride(1),
        alpha=alpha,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        MAX_SEQ_LEN=N,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(N),
        DimQ=DimQ,
        DimV=DimV,
        DeltaSize=DeltaSize,
        contextual_seq_len=contextual_seq_len,
        max_attn_len=max_attn_len,
        CAUSAL=causal,
        HAS_MULTIPLE_TARGETS=has_multiple_targets,
        IS_DELTA_Q=IS_DELTA_Q,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_CONTEXTUAL_SEQ_LEN=has_contextual_seq_len,
        HAS_MAX_ATTN_LEN=has_max_attn_len,
        HAS_SORT_BY_LENGTH_INDICES=has_sort_by_length_indices,
        **config,
    )

    return out


def triton_hstu_attention_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    N: int,
    alpha: float,
    max_attn_len: int,
    causal: float,
    contextual_seq_len: int,
    sort_by_length_indices: Optional[torch.Tensor],
    config: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes HSTU attention bwd pass.

    Key parameters:
    - dout: tensor with shape (L, H, D)
    - q: tensor with shape (L, H, D), L are sum of lengths of all sequences
    - k: tensor with shape (L, H, D), L are sum of lengths of all sequences
    - v: tensor with shape (L, H, D), L are sum of lengths of all sequences
    - dq: tensor with shape (L, H, D), gradients of q
    - dk: tensor with shape (L, H, D), gradients of k
    - dv: tensor with shape (L, H, D), gradients of v
    - seq_offsets: tensor with shape (B + 1), indicates lengths of each sequences.
    - num_targets: number of targets.
    - N: max sequence length
    - alpha: scale parameter to multiply output of first dot
    - max_attn_len: max attn length
    - causal: whether use causal mask.
    - contextual_seq_len: contexual sequence length.
    - sort_by_length_indices: indices of sequences sorted by lengths
    - config: Optional, tuning configs to run the kernel

    Returns:
    - dq, dk, dv: gradients of q, k, and v
    """
    _LOGGER.info(
        f"HSTU_ATTENTION_BKWD: dout={dout.shape}  q={tuple(q.shape)} k={tuple(k.shape)}  v={tuple(v.shape)} dq={tuple(dq.shape)} dk={tuple(dk.shape)}  dv={tuple(dv.shape)}"
    )
    dout = switch_to_contiguous_if_needed(dout)
    dq = switch_to_contiguous_if_needed(dq)
    dk = switch_to_contiguous_if_needed(dk)
    dv = switch_to_contiguous_if_needed(dv)
    if dout.shape[0] == 0:
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    Z = seq_offsets.numel() - 1
    _, H, DimQ = q.shape
    _, _, DimV = v.shape

    max_seq_len = autotune_max_seq_len(N)
    AUTOTUNE_Z = prev_power_of_2(Z)
    if config is None:
        config = _get_bwd_config(AUTOTUNE_Z, H, max_seq_len, DimQ, DimV)

    grid = lambda meta: (  # noqa E731
        Z * H,
        (triton.cdiv(N, meta["BLOCK_N"]) if meta["SEQUENCE_PARALLEL"] else 1),
    )
    # The minimum size of BLOCK_M used in `_get_bw_configs`.
    # TODO (linjianma): avoid hardcoding the value.
    MIN_BLOCK_M = 16
    lock = torch.empty(
        (Z * H, triton.cdiv(N, MIN_BLOCK_M)),
        dtype=torch.int32,
        device=q.device,
    )

    dq.zero_()
    if config["SEQUENCE_PARALLEL"] == 1:
        lock.zero_()

    _hstu_attn_bwd[grid](
        Q=q,
        K=k,
        V=v,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        DOut=dout,
        DQ=dq,
        DK=dk,
        DV=dv,
        LOCK=lock,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_dom=dout.stride(0),
        stride_doh=dout.stride(1),
        stride_dqm=dq.stride(0),
        stride_dqh=dq.stride(1),
        stride_dkn=dk.stride(0),
        stride_dkh=dk.stride(1),
        stride_dvn=dv.stride(0),
        stride_dvh=dv.stride(1),
        alpha=alpha,
        contextual_seq_len=contextual_seq_len,
        max_attn_len=max_attn_len,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        MAX_SEQ_LEN=N,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(N),
        DimQ=DimQ,
        DimV=DimV,
        CAUSAL=causal,
        HAS_MULTIPLE_TARGETS=num_targets is not None,
        HAS_CONTEXTUAL_SEQ_LEN=contextual_seq_len > 0,
        HAS_MAX_ATTN_LEN=max_attn_len > 0,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_SORT_BY_LENGTH_INDICES=sort_by_length_indices is not None,
        **config,
    )

    return dq, dk, dv


class _AttentionFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        N: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
        causal: bool,
        num_targets: Optional[torch.Tensor],
        max_attn_len: int,
        contextual_seq_len: int,
        sort_by_length: bool,
    ) -> torch.Tensor:
        sort_by_length_indices = None
        if sort_by_length:
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            _, sort_by_length_indices = torch.sort(
                seq_lengths, descending=True, stable=False
            )
        saved_tensors = [q, k, v, seq_offsets]
        if num_targets is not None:
            saved_tensors.append(num_targets)
        if sort_by_length_indices is not None:
            saved_tensors.append(sort_by_length_indices)
        ctx.save_for_backward(*saved_tensors)
        ctx.alpha = alpha
        ctx.causal = causal
        ctx.has_multiple_targets = num_targets is not None
        ctx.max_attn_len = max_attn_len
        ctx.N = N
        ctx.contextual_seq_len = contextual_seq_len
        ctx.sort_by_length = sort_by_length
        return triton_hstu_attention_fwd(
            N=N,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=causal,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length_indices=sort_by_length_indices,
        )

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, dout: torch.Tensor) -> Tuple[
        None,
        None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        with torch.inference_mode():
            q, k, v, seq_offsets = ctx.saved_tensors[:4]
            idx = 4
            if ctx.has_multiple_targets:
                num_targets = ctx.saved_tensors[idx]
                idx += 1
            else:
                num_targets = None
            if ctx.sort_by_length:
                sort_by_length_indices = ctx.saved_tensors[idx]
            else:
                sort_by_length_indices = None

            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            dq, dk, dv = triton_hstu_attention_bwd(
                dout=dout,
                q=q,
                k=k,
                v=v,
                dq=dq,
                dk=dk,
                dv=dv,
                seq_offsets=seq_offsets,
                num_targets=num_targets,
                N=ctx.N,
                alpha=ctx.alpha,
                max_attn_len=ctx.max_attn_len,
                causal=ctx.causal,
                contextual_seq_len=ctx.contextual_seq_len,
                sort_by_length_indices=sort_by_length_indices,
            )
            return (
                None,
                None,
                dq,
                dk,
                dv,
                None,
                None,
                None,
                None,
                None,
                None,
            )
