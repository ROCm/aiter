# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""KDA gating recurrence on gfx1250: decode, state fork and speculative verify.

``kda_gating_decode`` runs the per-channel KDA gate
``lower_bound * sigmoid(exp(A_log) * (a + dt_bias))`` and the delta-rule
update in one Gluon kernel. A read-slot table apart from the write-slot table
lets a sequence read its state from one slot and write it to another slot in
the same launch.

``supports`` checks the setup that a model fixes before its first call, so a
caller can select its route one time.
"""

import torch

from aiter.ops.triton._gluon_kernels.gfx1250.attention.kda_gating import (
    get_kda_gating_config,
    kda_gating_kernel,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.fast_launch import fast_launch
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
_launch = fast_launch(kda_gating_kernel)
_ACT_DTYPES = (torch.bfloat16, torch.float16)
_HEAD_DIM = 128


def supports(
    *,
    lower_bound: float | None,
    state_dtype: torch.dtype,
    head_k_dim: int,
    head_v_dim: int,
    act_dtype: torch.dtype,
) -> bool:
    """Whether ``kda_gating_decode`` takes a model with this setup on this GPU.

    The kernel needs the lower-bounded gate, an FP32 state, ``head_k_dim =
    head_v_dim = 128``, BF16 or FP16 activations and a gfx1250 GPU. Its config
    tables are keyed by the exact arch, as all tuned configs are, and only
    gfx1250 has them, so a gfx1250-strict GPU is not supported.
    ``kda_gating_decode`` checks the other limits on each call.
    """
    return (
        lower_bound is not None
        and state_dtype == torch.float32
        and head_k_dim == _HEAD_DIM
        and head_v_dim == _HEAD_DIM
        and act_dtype in _ACT_DTYPES
        and arch_info.get_arch() == "gfx1250"
    )


def _decline(reason: str, *args) -> None:
    _LOGGER.debug("kda_gating_decode declines the call: " + reason, *args)


def kda_gating_decode(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.IntTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    ssm_state_indices_in: torch.Tensor | None = None,
    lower_bound: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run the KDA gating recurrence, or return None without a launch.

    ``q`` and ``k`` are ``[1, T, H, K]`` and ``v`` is ``[1, T, HV, V]``, with
    HV a multiple of H. ``a`` is the gate input ``[1, T, HV, K]`` and ``b`` the
    beta logit ``[1, T, HV]``: ``beta = sigmoid(b)``. The activations have one
    dtype and a unit stride in the last dimension, and q, k, v and a are dense
    in their last two dimensions. ``A_log`` has HV values and ``dt_bias``
    HV * K, both contiguous.

    ``initial_state`` is the FP32 state pool ``[slots, HV, V, K]``. Each slot
    is dense, and the slot stride can have any value. The kernel updates the
    pool in place and L2-normalizes q and k, so it needs
    ``inplace_final_state`` and ``use_qk_l2norm_in_kernel``.

    ``cu_seqlens`` is int32. ``ssm_state_indices`` is the int32 write-slot
    table. ``ssm_state_indices_in`` is the read-slot table, with the shape and
    strides of the write-slot table. None reads from the write slots. A 1D
    table has one slot for each sequence, and each sequence has 0 or 1 tokens.
    A 2D table has a row for each sequence and a column for each token, and a
    sequence has at most one token for each column. The read slot is
    ``read[seq, num_accepted_tokens[seq] - 1]`` in spec verify (int32
    ``num_accepted_tokens`` and a 2D table), else ``read[seq]`` or
    ``read[seq, 0]``. In spec verify a sequence with tokens needs a count in
    ``[1, columns]``. The kernel cannot check these lengths and counts, because
    they are on the device.

    A read slot below 0, an empty sequence or a count below 1 leaves the output
    rows and the slots of that sequence unchanged. A write slot below 0 still
    writes the output. Slot 0 is a real slot.

    ``o`` is ``[T, HV, V]`` or ``v``'s shape, with ``v``'s dtype, and dense in
    ``[HV, V]``. With ``o=None`` the kernel writes to a new tensor with ``v``'s
    shape, and the rows that it skips are not initialized.

    Returns:
        ``(o, initial_state)``, or None when the kernel does not take the
        inputs. The DEBUG log of ``AITER_TRITON`` gives the reason.
    """
    if initial_state is None or cu_seqlens is None or ssm_state_indices is None:
        return _decline("initial_state, cu_seqlens and ssm_state_indices are needed")
    if not (inplace_final_state and use_qk_l2norm_in_kernel):
        return _decline(
            "inplace_final_state=%s and use_qk_l2norm_in_kernel=%s, both must be True",
            inplace_final_state,
            use_qk_l2norm_in_kernel,
        )
    if q.ndim != 4 or q.shape[0] != 1 or v.ndim != 4:
        return _decline(
            "q is %s and v is %s, both must be [1, T, heads, dim]",
            tuple(q.shape),
            tuple(v.shape),
        )
    _, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    state = initial_state
    if not supports(
        lower_bound=lower_bound,
        state_dtype=state.dtype,
        head_k_dim=K,
        head_v_dim=V,
        act_dtype=q.dtype,
    ):
        return _decline(
            "supports() is False for lower_bound %s, a %s state, K %d, V %d and "
            "%s activations on %s",
            lower_bound,
            state.dtype,
            K,
            V,
            q.dtype,
            arch_info.get_arch(),
        )
    if HV % H:
        return _decline("HV %d is not a multiple of H %d", HV, H)
    if not (
        k.shape == q.shape
        and v.shape[:2] == q.shape[:2]
        and a.shape == (1, T, HV, K)
        and b.shape == (1, T, HV)
        and all(x.dtype == q.dtype for x in (k, v, a, b))
    ):
        return _decline("the shape or dtype of k, v, a or b does not agree with q")
    if not (
        q.stride()[2:] == (K, 1)
        and k.stride()[2:] == (K, 1)
        and v.stride()[2:] == (V, 1)
        and a.stride()[2:] == (K, 1)
        and b.stride(2) == 1
    ):
        return _decline("q, k, v, a or b is not dense in its last dimensions")
    if not (
        state.ndim == 4
        and state.shape[1:] == (HV, V, K)
        and state.stride()[1:] == (V * K, K, 1)
    ):
        return _decline(
            "initial_state is %s with strides %s, it must have dense [HV, V, K] slots",
            tuple(state.shape),
            state.stride(),
        )
    if not (
        A_log.numel() == HV
        and A_log.is_contiguous()
        and dt_bias.numel() == HV * K
        and dt_bias.is_contiguous()
    ):
        return _decline("A_log must have HV and dt_bias HV * K contiguous values")
    if not (
        cu_seqlens.dtype == torch.int32
        and cu_seqlens.ndim == 1
        and cu_seqlens.is_contiguous()
    ):
        return _decline("cu_seqlens must be a contiguous 1D int32 tensor")
    write = ssm_state_indices
    read = write if ssm_state_indices_in is None else ssm_state_indices_in
    if not (
        write.dtype == torch.int32
        and read.dtype == torch.int32
        and write.ndim in (1, 2)
        and write.stride(-1) == 1
    ):
        return _decline("the slot tables must be 1D or 2D int32 with unit last stride")
    if not (read.shape == write.shape and read.stride() == write.stride()):
        return _decline(
            "ssm_state_indices_in must have the shape and strides of ssm_state_indices"
        )
    multi_token = write.ndim == 2
    spec = num_accepted_tokens is not None
    n_seq = cu_seqlens.numel() - 1
    if not multi_token and T > n_seq:
        return _decline(
            "a 1D slot table gives each sequence at most 1 token, but T %d > %d "
            "sequences",
            T,
            n_seq,
        )
    if not (write.shape[0] >= n_seq and (not multi_token or write.shape[1] > 0)):
        return _decline(
            "the slot tables need a row for each sequence and at least one column"
        )
    if spec and not (
        multi_token
        and num_accepted_tokens.dtype == torch.int32
        and num_accepted_tokens.is_contiguous()
        and num_accepted_tokens.numel() >= n_seq
    ):
        return _decline(
            "spec verify needs 2D slot tables and contiguous int32 "
            "num_accepted_tokens with an entry for each sequence"
        )
    tensors = (
        A_log,
        a,
        b,
        dt_bias,
        k,
        v,
        o,
        state,
        cu_seqlens,
        write,
        read,
        num_accepted_tokens,
    )
    if not (q.is_cuda and all(x is None or x.device == q.device for x in tensors)):
        return _decline("all the tensors must be on the CUDA device of q")
    if o is None:
        o = torch.empty_like(v)
        stride_o_tok = o.stride(1)
    elif o.dtype != v.dtype or o.stride()[-2:] != (V, 1):
        return _decline("o must have the dtype of v and be dense in [HV, V]")
    elif o.shape == v.shape:
        stride_o_tok = o.stride(1)
    elif o.shape == v.shape[1:]:
        stride_o_tok = o.stride(0)
    else:
        return _decline(
            "o is %s, it must be %s or %s",
            tuple(o.shape),
            tuple(v.shape),
            tuple(v.shape[1:]),
        )
    if n_seq <= 0:
        return o, state
    if scale is None:
        scale = K**-0.5
    config = get_kda_gating_config(n_seq * HV, multi_token)
    num_warps = config.pop("num_warps")
    _launch[(n_seq * HV * (V // config["BV"]),)](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        a_ptr=a,
        b_ptr=b,
        A_log_ptr=A_log,
        dt_bias_ptr=dt_bias,
        o_ptr=o,
        state_ptr=state,
        cu_ptr=cu_seqlens,
        write_ptr=write,
        read_ptr=read,
        accepted_ptr=num_accepted_tokens if spec else cu_seqlens,
        stride_idx_seq=write.stride(0),
        stride_q_tok=q.stride(1),
        stride_k_tok=k.stride(1),
        stride_v_tok=v.stride(1),
        stride_a_tok=a.stride(1),
        stride_b_tok=b.stride(1),
        stride_o_tok=stride_o_tok,
        stride_slot=state.stride(0),
        lower_bound=float(lower_bound),
        scale=float(scale),
        H=H,
        HV=HV,
        K=K,
        V=V,
        NUM_WARPS=num_warps,
        MULTI_TOKEN=multi_token,
        IS_SPEC=spec,
        **config,
        num_warps=num_warps,
        enable_fp_fusion=True,
    )
    return o, state
