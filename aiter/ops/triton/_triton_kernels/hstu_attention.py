# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2024, The vLLM team.
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
import json

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
import functools
from ..utils._triton.pid_preprocessing import pid_grid, remap_xcd
from ..utils._triton import arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH

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


@triton.jit
def _hstu_attn_fwd_one_block(  # noqa: C901
    start_n,
    seq_len,
    offs_m,
    offs_n,
    q,
    K_block_ptr,
    V_block_ptr,
    n_targets,
    alpha,
    MAX_SEQ_LEN,
    contextual_seq_len,
    max_attn_len,
    CAUSAL: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    # -- compute qk ----
    k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
    qk = tl.dot(q, k, allow_tf32=ALLOW_TF32) * alpha
    invalid_mask = offs_m[:, None] == offs_n[None, :]
    max_ids = seq_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        offs_m = offs_m - contextual_seq_len + 1
        offs_m = tl.where(
            offs_m > 0,
            offs_m,
            0,
        )
        offs_n = offs_n - contextual_seq_len + 1
        offs_n = tl.where(
            offs_n > 0,
            offs_n,
            0,
        )
        max_ids = max_ids - contextual_seq_len + 1
    if HAS_MULTIPLE_TARGETS:
        max_ids = max_ids - n_targets
        offs_m = tl.where(
            offs_m < max_ids,
            offs_m,
            max_ids,
        )
        offs_n = tl.where(
            offs_n < max_ids,
            offs_n,
            max_ids,
        )
    offs_m_minus_n = offs_m[:, None] - offs_n[None, :]
    if not CAUSAL:
        offs_m_minus_n = tl.where(offs_m_minus_n > 0, offs_m_minus_n, -offs_m_minus_n)
    invalid_mask = invalid_mask | (offs_m_minus_n > 0)
    if HAS_MAX_ATTN_LEN:
        invalid_mask = invalid_mask and offs_m_minus_n <= max_attn_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        invalid_mask = invalid_mask or (
            offs_m[:, None] == 0 and offs_n[None, :] < max_ids
        )
    # pyre-fixme[16]: Module `math` has no attribute `fast_dividef`.
    silu = fast_dividef(qk, 1.0 + fast_expf(-qk)) * (1.0 / MAX_SEQ_LEN)
    silu = tl.where(invalid_mask, silu, 0)
    v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
    silu = silu.to(v.dtype)
    return tl.dot(silu, v, allow_tf32=ALLOW_TF32)


@triton.jit
def _hstu_attn_fwd_compute(  # noqa C901
    Q,
    K,
    V,
    seq_offsets,
    num_targets,
    Out,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    alpha,
    MAX_SEQ_LEN,
    DeltaSize,
    contextual_seq_len,
    max_attn_len,
    off_z,
    off_h,
    pid,
    CAUSAL: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
):
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    off_h = off_h.to(tl.int64)
    off_z = off_z.to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    if IS_DELTA_Q:
        start_m_delta = pid * BLOCK_M
        start_m = (start_m_delta + seq_len - DeltaSize).to(tl.int32)
    else:
        start_m_delta = 0
        start_m = pid * BLOCK_M
    if start_m < seq_len:
        if HAS_MULTIPLE_TARGETS:
            n_targets = tl.load(num_targets + off_z).to(tl.int32)
        else:
            n_targets = None

        # initialize offsets
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        if IS_DELTA_Q:
            Q_block_ptr = tl.make_block_ptr(
                base=Q + off_h * stride_qh + off_z * DeltaSize * stride_qm,
                shape=(DeltaSize, BLOCK_D_Q),
                strides=(stride_qm, 1),
                offsets=(start_m_delta, 0),
                block_shape=(BLOCK_M, BLOCK_D_Q),
                order=(1, 0),
            )
        else:
            Q_block_ptr = tl.make_block_ptr(
                base=Q + off_h * stride_qh + seq_start * stride_qm,
                shape=(seq_len, BLOCK_D_Q),
                strides=(stride_qm, 1),
                offsets=(start_m, 0),
                block_shape=(BLOCK_M, BLOCK_D_Q),
                order=(1, 0),
            )
        K_block_ptr = tl.make_block_ptr(
            base=K + off_h * stride_kh + seq_start * stride_kn,
            shape=(BLOCK_D_Q, seq_len),
            strides=(1, stride_kn),
            offsets=(0, 0),
            block_shape=(BLOCK_D_Q, BLOCK_N),
            order=(0, 1),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + off_h * stride_vh + seq_start * stride_vn,
            shape=(seq_len, BLOCK_D_V),
            strides=(stride_vn, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_D_V),
            order=(1, 0),
        )

        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
        acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)
        if CAUSAL:
            if HAS_MULTIPLE_TARGETS:
                uih_end = seq_len - n_targets
            else:
                uih_end = seq_len
            if HAS_CONTEXTUAL_SEQ_LEN is True and start_m < contextual_seq_len:
                # uih_end must be larger than start_m
                low = 0
                high = seq_len
            else:
                low = 0
                high = start_m + BLOCK_M
                if HAS_MAX_ATTN_LEN:
                    if start_m > uih_end:
                        low = uih_end - max_attn_len
                    else:
                        low = start_m - max_attn_len
                    if HAS_CONTEXTUAL_SEQ_LEN:
                        low = low if low > contextual_seq_len else 0
                    else:
                        low = low if low > 0 else 0
                if HAS_MULTIPLE_TARGETS:
                    uih_end = (uih_end + BLOCK_N - 1) // BLOCK_N * BLOCK_N
                    if uih_end < start_m:
                        high = seq_len - n_targets
        else:
            low = 0
            high = seq_len

        if low > 0:
            K_block_ptr = tl.advance(K_block_ptr, (0, low))
            V_block_ptr = tl.advance(V_block_ptr, (low, 0))
        end_n = low
        for start_n in range(low, high, BLOCK_N):
            acc += _hstu_attn_fwd_one_block(
                start_n=start_n,
                seq_len=seq_len,
                offs_m=offs_m,
                offs_n=offs_n + start_n,
                q=q,
                K_block_ptr=K_block_ptr,
                V_block_ptr=V_block_ptr,
                n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len,
                CAUSAL=CAUSAL,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_N=BLOCK_N,
            )
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            end_n += BLOCK_N

        if HAS_MULTIPLE_TARGETS and CAUSAL:
            # pyre-ignore[61]
            if uih_end < start_m:
                low_delta = start_m
                high_delta = start_m + BLOCK_M
                offset = (low_delta - end_n).to(tl.int32)
                K_block_ptr = tl.advance(K_block_ptr, (0, offset))
                V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
                for start_delta in tl.range(
                    low_delta, high_delta, BLOCK_N, num_stages=0
                ):
                    acc += _hstu_attn_fwd_one_block(
                        start_n=start_delta,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        offs_n=offs_n + start_delta,
                        q=q,
                        K_block_ptr=K_block_ptr,
                        V_block_ptr=V_block_ptr,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        alpha=alpha,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        contextual_seq_len=contextual_seq_len,
                        max_attn_len=max_attn_len,
                        CAUSAL=CAUSAL,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_N=BLOCK_N,
                    )
                    K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
                    V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

        if IS_DELTA_Q:
            start_m_delta = pid * BLOCK_M
            offs_m_delta = start_m_delta + tl.arange(0, BLOCK_M)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            off_o = Out + off_z * DeltaSize * stride_om + off_h * stride_oh
            out_ptrs = off_o + offs_m_delta[:, None] * stride_om + offs_v_d[None, :]
            tl.store(out_ptrs, acc, mask=(offs_m_delta < DeltaSize)[:, None])
        else:
            # rematerialize offsets to save registers
            start_m = pid * BLOCK_M
            offs_m = start_m + tl.arange(0, BLOCK_M)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            off_o = Out + seq_start * stride_om + off_h * stride_oh
            out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[None, :]
            tl.store(out_ptrs, acc, mask=(offs_m < seq_len)[:, None])


@triton.jit
def _hstu_attn_fwd(  # noqa C901
    Q,
    K,
    V,
    sort_by_length_indices,
    seq_offsets,
    num_targets,
    Out,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    alpha,
    Z,
    AUTOTUNE_Z,
    H,
    MAX_SEQ_LEN,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ,
    DimV,
    DeltaSize,
    contextual_seq_len,
    max_attn_len,
    CAUSAL: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_SORT_BY_LENGTH_INDICES: tl.constexpr,
):
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h = off_hz % H
    pid = tl.program_id(0)
    _hstu_attn_fwd_compute(
        Q=Q,
        K=K,
        V=V,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        Out=Out,
        stride_qm=stride_qm,
        stride_qh=stride_qh,
        stride_kn=stride_kn,
        stride_kh=stride_kh,
        stride_vn=stride_vn,
        stride_vh=stride_vh,
        stride_om=stride_om,
        stride_oh=stride_oh,
        alpha=alpha,
        MAX_SEQ_LEN=MAX_SEQ_LEN,
        DeltaSize=DeltaSize,
        contextual_seq_len=contextual_seq_len,
        max_attn_len=max_attn_len,
        off_z=off_z,
        off_h=off_h,
        pid=pid,
        CAUSAL=CAUSAL,
        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
        IS_DELTA_Q=IS_DELTA_Q,
        ALLOW_TF32=ALLOW_TF32,
        BLOCK_D_Q=BLOCK_D_Q,
        BLOCK_D_V=BLOCK_D_V,
        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )


@triton.jit
def _hstu_attn_bwd_one_block(  # noqa C901
    start_m,
    offs_n,
    offs_m,
    q_ptrs_trans,
    dq_ptrs_trans,
    mask_n,
    do_ptrs,
    dk,
    dv,
    k,
    v,
    pos_offs_n,
    seq_len,
    n_targets,
    max_ids,
    contextual_seq_len,
    max_attn_len,
    LOCK,
    stride_qm,
    stride_dom,
    stride_dqm,
    alpha,
    MAX_SEQ_LEN,
    CAUSAL: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
):
    pos_offs_m = offs_m + start_m
    mask_m = pos_offs_m < seq_len
    invalid_mask_trans = pos_offs_m[None, :] == offs_n[:, None]
    # recompute qk and silu
    if HAS_CONTEXTUAL_SEQ_LEN:
        pos_offs_m = pos_offs_m - contextual_seq_len + 1
        pos_offs_m = tl.where(
            pos_offs_m > 0,
            pos_offs_m,
            0,
        )
    if HAS_MULTIPLE_TARGETS:
        pos_offs_m = tl.where(
            pos_offs_m < max_ids,
            pos_offs_m,
            max_ids,
        )
    q_trans = tl.load(
        q_ptrs_trans + start_m * stride_qm,
        mask=mask_m[None, :],
        other=0.0,
    )
    qk_trans = tl.dot(k, q_trans, allow_tf32=ALLOW_TF32) * alpha
    # pyre-fixme[16]: Module `math` has no attribute `fast_dividef`.
    sig_trans = fast_dividef(1.0, 1.0 + tl.exp(-qk_trans))
    silu_trans = qk_trans * sig_trans * (1.0 / MAX_SEQ_LEN)
    pos_offs_m_minus_n = pos_offs_m[None, :] - pos_offs_n[:, None]
    if not CAUSAL:
        pos_offs_m_minus_n = tl.where(
            pos_offs_m_minus_n > 0, pos_offs_m_minus_n, -pos_offs_m_minus_n
        )
    invalid_mask_trans = invalid_mask_trans | (pos_offs_m_minus_n > 0)
    if HAS_MAX_ATTN_LEN:
        invalid_mask_trans = invalid_mask_trans and pos_offs_m_minus_n <= max_attn_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        invalid_mask_trans = invalid_mask_trans or (
            pos_offs_m[None, :] == 0 and pos_offs_n[:, None] < max_ids
        )
    silu_trans = tl.where(invalid_mask_trans, silu_trans, 0)
    silu_trans = silu_trans.to(k.dtype)
    # compute dv
    do = tl.load(
        do_ptrs + start_m * stride_dom,
        mask=mask_m[:, None],
        other=0.0,
    )
    dv += tl.dot(silu_trans, do, allow_tf32=ALLOW_TF32)

    # compute dk and dq
    dqk_trans = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32)
    dqk_trans = (
        dqk_trans * sig_trans * (1 + qk_trans * (1 - sig_trans)) * (1.0 / MAX_SEQ_LEN)
    )
    dqk_trans = tl.where(invalid_mask_trans, dqk_trans, 0)
    dqk_trans = dqk_trans.to(k.dtype)

    # Note: the factor `alpha` is delayed until the end of the function to reduce the cost
    dk += tl.dot(dqk_trans, tl.trans(q_trans), allow_tf32=ALLOW_TF32)
    if ATOMIC_ADD:
        lock_id = start_m // BLOCK_M
        stride_lock = tl.cdiv(MAX_SEQ_LEN, BLOCK_M)
        lock = LOCK + tl.program_id(0) * stride_lock + lock_id
        tl.debug_barrier()  # add a barrier to force sync
        while tl.atomic_cas(lock, 0, 1) == 1:
            pass
    dq_trans = tl.load(
        dq_ptrs_trans + start_m * stride_dqm,
        mask=mask_m[None, :],
        other=0.0,
        eviction_policy="evict_last",
    )
    dq_trans += tl.dot(tl.trans(k), dqk_trans, allow_tf32=ALLOW_TF32) * alpha
    dq_trans = dq_trans.to(k.dtype)
    tl.store(
        dq_ptrs_trans + start_m * stride_dqm,
        dq_trans,
        mask=mask_m[None, :],
        eviction_policy="evict_last",
    )
    if ATOMIC_ADD:
        tl.atomic_xchg(lock, 0)  # pyre-ignore [61]
    return dk, dv


@triton.jit
def _hstu_attn_bwd_one_col_block(  # noqa C901
    start_n,
    seq_len,
    n_targets,
    contextual_seq_len,
    max_attn_len,
    Q,
    K,
    V,
    DOut,
    DQ,
    DK,
    DV,
    LOCK,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    alpha,
    MAX_SEQ_LEN,
    CAUSAL: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
):
    # Work on the subsequence dv[start_n, start_n + BLOCK_N, :]
    if CAUSAL:
        if HAS_MULTIPLE_TARGETS:
            low = start_n
            if HAS_MAX_ATTN_LEN:
                high = start_n + max_attn_len + BLOCK_N
                high = high if high + n_targets < seq_len else seq_len
            else:
                high = seq_len
        else:
            low = start_n
            if HAS_MAX_ATTN_LEN:
                high = start_n + max_attn_len + BLOCK_N
                high = high if high < seq_len else seq_len
            else:
                high = seq_len
        if HAS_CONTEXTUAL_SEQ_LEN:
            contextual_block_end = tl.cdiv(contextual_seq_len, BLOCK_M) * BLOCK_M
            if low < contextual_block_end:
                low = contextual_block_end
    else:
        low = 0
        high = start_n + BLOCK_N

    # initialize row/col offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_qk_d = tl.arange(0, BLOCK_D_Q)
    offs_v_d = tl.arange(0, BLOCK_D_V)
    offs_n = start_n + tl.arange(0, BLOCK_N)

    # initialize pointers to value-like data
    q_ptrs_trans = Q + (offs_m[None, :] * stride_qm + offs_qk_d[:, None])
    dq_ptrs_trans = DQ + (offs_m[None, :] * stride_dqm + offs_qk_d[:, None])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_qk_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_v_d[None, :])
    mask_n = offs_n < seq_len

    do_ptrs = DOut + (offs_m[:, None] * stride_dom + offs_v_d[None, :])
    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
    # k and v stay in SRAM throughout
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    max_ids = seq_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        pos_offs_n = offs_n - contextual_seq_len + 1
        pos_offs_n = tl.where(
            pos_offs_n > 0,
            pos_offs_n,
            0,
        )
        max_ids = max_ids - contextual_seq_len + 1
    else:
        pos_offs_n = offs_n
    if HAS_MULTIPLE_TARGETS:
        max_ids = max_ids - n_targets
        pos_offs_n = tl.where(
            pos_offs_n < max_ids,
            pos_offs_n,
            max_ids,
        )
    # loop over rows
    if HAS_CONTEXTUAL_SEQ_LEN and CAUSAL:
        for start_m in range(0, contextual_seq_len, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            dk, dv = _hstu_attn_bwd_one_block(
                start_m=start_m,
                offs_n=offs_n,
                offs_m=offs_m,
                q_ptrs_trans=q_ptrs_trans,
                dq_ptrs_trans=dq_ptrs_trans,
                mask_n=mask_n,
                do_ptrs=do_ptrs,
                dk=dk,
                dv=dv,
                k=k,
                v=v,
                pos_offs_n=pos_offs_n,
                seq_len=seq_len,
                n_targets=n_targets,
                max_ids=max_ids,
                contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len,
                LOCK=LOCK,
                stride_qm=stride_qm,
                stride_dom=stride_dom,
                stride_dqm=stride_dqm,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                CAUSAL=CAUSAL,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                ATOMIC_ADD=ATOMIC_ADD,
            )
    for start_m in tl.range(low, high, BLOCK_M, loop_unroll_factor=UNROLL):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        dk, dv = _hstu_attn_bwd_one_block(
            start_m=start_m,
            offs_n=offs_n,
            offs_m=offs_m,
            q_ptrs_trans=q_ptrs_trans,
            dq_ptrs_trans=dq_ptrs_trans,
            mask_n=mask_n,
            do_ptrs=do_ptrs,
            dk=dk,
            dv=dv,
            k=k,
            v=v,
            pos_offs_n=pos_offs_n,
            seq_len=seq_len,
            n_targets=n_targets,
            max_ids=max_ids,
            contextual_seq_len=contextual_seq_len,
            max_attn_len=max_attn_len,
            LOCK=LOCK,
            stride_qm=stride_qm,
            stride_dom=stride_dom,
            stride_dqm=stride_dqm,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            CAUSAL=CAUSAL,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            ATOMIC_ADD=ATOMIC_ADD,
        )
    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
    dk = dk * alpha
    tl.store(dv_ptrs, dv.to(k.dtype), mask=mask_n[:, None])
    tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])


@triton.jit
def _hstu_attn_bwd(  # noqa C901
    Q,
    K,
    V,
    sort_by_length_indices,
    seq_offsets,
    num_targets,
    DOut,
    DQ,
    DK,
    DV,
    LOCK,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    alpha,
    contextual_seq_len,
    max_attn_len,
    Z,
    AUTOTUNE_Z,
    H,
    MAX_SEQ_LEN,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ,
    DimV,
    CAUSAL: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
    HAS_SORT_BY_LENGTH_INDICES: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h = off_hz % H
    off_h = off_h.to(tl.int64)
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    if HAS_MULTIPLE_TARGETS:
        n_targets = tl.load(num_targets + off_z).to(tl.int32)
    else:
        n_targets = None
    # offset pointers for batch/head
    Q = Q + seq_start * stride_qm + off_h * stride_qh
    K = K + seq_start * stride_kn + off_h * stride_kh
    V = V + seq_start * stride_vn + off_h * stride_vh
    DOut = DOut + seq_start * stride_dom + off_h * stride_doh
    DQ = DQ + seq_start * stride_dqm + off_h * stride_dqh
    DK = DK + seq_start * stride_dkn + off_h * stride_dkh
    DV = DV + seq_start * stride_dvn + off_h * stride_dvh
    if SEQUENCE_PARALLEL:
        start_n = tl.program_id(1) * BLOCK_N
        if start_n >= seq_len:
            return
        _hstu_attn_bwd_one_col_block(
            start_n=start_n,
            seq_len=seq_len,
            n_targets=n_targets,
            contextual_seq_len=contextual_seq_len,
            max_attn_len=max_attn_len,
            Q=Q,
            K=K,
            V=V,
            DOut=DOut,
            DQ=DQ,
            DK=DK,
            DV=DV,
            LOCK=LOCK,
            stride_qm=stride_qm,
            stride_kn=stride_kn,
            stride_vn=stride_vn,
            stride_dom=stride_dom,
            stride_dqm=stride_dqm,
            stride_dkn=stride_dkn,
            stride_dvn=stride_dvn,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            CAUSAL=CAUSAL,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            UNROLL=UNROLL,
            ATOMIC_ADD=True,
        )
    else:
        for start_n in range(0, seq_len, BLOCK_N):
            _hstu_attn_bwd_one_col_block(
                start_n=start_n,
                seq_len=seq_len,
                n_targets=n_targets,
                contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len,
                Q=Q,
                K=K,
                V=V,
                DOut=DOut,
                DQ=DQ,
                DK=DK,
                DV=DV,
                LOCK=LOCK,
                stride_qm=stride_qm,
                stride_kn=stride_kn,
                stride_vn=stride_vn,
                stride_dom=stride_dom,
                stride_dqm=stride_dqm,
                stride_dkn=stride_dkn,
                stride_dvn=stride_dvn,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                CAUSAL=CAUSAL,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_D_Q=BLOCK_D_Q,
                BLOCK_D_V=BLOCK_D_V,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                UNROLL=UNROLL,
                ATOMIC_ADD=False,
            )


@functools.lru_cache(maxsize=1024)
def _get_fwd_config(
    AUTOTUNE_Z: int,
    H: int,
    AUTOTUNE_MAX_SEQ_LEN: int,
    DimQ: int,
    DimV: int,
    DeltaSize: int,
    IS_DELTA_Q: bool,
):
    if not hasattr(_get_fwd_config, "_config_dict"):
        dev = arch_info.get_device()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/hstu_attn/{dev}-HSTU_ATTN_FWD.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_fwd_config._config_dict = config

    if AUTOTUNE_Z < 512:
        batch_key = "small_batch"
    elif AUTOTUNE_Z == 512:
        batch_key = "batch_512"
    else:
        batch_key = "large_batch"

    return _get_fwd_config._config_dict[batch_key]


@functools.lru_cache(maxsize=1024)
def _get_bwd_config(
    AUTOTUNE_Z: int,
    H: int,
    AUTOTUNE_MAX_SEQ_LEN: int,
    DimQ: int,
    DimV: int,
):
    if not hasattr(_get_bwd_config, "_config_dict"):
        dev = arch_info.get_device()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/hstu_attn/{dev}-HSTU_ATTN_BWD.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_bwd_config._config_dict = config

    if AUTOTUNE_Z < 512:
        batch_key = "small_batch"
    else:
        batch_key = "large_batch"

    return _get_bwd_config._config_dict[batch_key]
