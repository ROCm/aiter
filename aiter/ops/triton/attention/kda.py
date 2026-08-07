# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import logging
import math
import warnings

import torch
import triton
from packaging.version import Version

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
_LOG_INFO = _LOGGER._logger.isEnabledFor(logging.INFO)

_TRITON_GE_36 = Version(triton.__version__.split("+")[0]) >= Version("3.6.0")
_ARCH = arch_info.get_arch()

fused_recurrent_kda_packed_decode_kernel = None
if _TRITON_GE_36 and _ARCH == "gfx1250":
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.kda_decode import (
        fused_recurrent_kda_packed_decode_kernel,
    )


def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    output_final_state: bool = True,
    inplace_final_state: bool = True,
    state_v_first: bool | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    lower_bound: float | None = None,
    out: torch.Tensor | None = None,
    state_out: torch.Tensor | None = None,
    BV: int = 32,
    SK: int | None = None,
    num_warps: int | None = None,
    num_buffers: int = 1,
    load_buffers: int | None = None,
    nt_stream: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:

    if fused_recurrent_kda_packed_decode_kernel is None:
        raise RuntimeError(
            f"kda gluon decode requires triton>=3.6.0 on gfx1250 "
            f"(found triton {triton.__version__} on {_ARCH})"
        )
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError(
            "allow_neg_eigval=True requires use_beta_sigmoid_in_kernel=True"
        )

    if "transpose_state_layout" in kwargs:
        if state_v_first is not None:
            raise ValueError(
                "Cannot pass both `state_v_first` and the deprecated "
                "`transpose_state_layout`."
            )
        warnings.warn(
            "`transpose_state_layout` is deprecated and renamed to `state_v_first`.",
            DeprecationWarning,
            stacklevel=2,
        )
        state_v_first = kwargs.pop("transpose_state_layout")
    if kwargs:
        raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
    if state_v_first is None:
        state_v_first = True

    if not (
        q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and g.is_contiguous()
        and beta.is_contiguous()
    ):
        q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))

    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    if scale is None:
        scale = K**-0.5

    assert k.shape == q.shape, "q and k must match"
    assert v.shape[:2] == q.shape[:2], "v must share [B, T] with q"
    assert g.shape == (B, T, HV, K), "g must be [B, T, HV, K]"
    assert HV % H == 0, "HV must be divisible by H"
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {B} when using "
                f"`cu_seqlens`. Please flatten variable-length inputs before processing."
            )
        if (
            initial_state is not None
            and ssm_state_indices is None
            and initial_state.shape[0] != N
        ):
            raise ValueError(
                f"The number of initial states is expected to equal the number of "
                f"input sequences, i.e. {N} rather than {initial_state.shape[0]}."
            )
    if use_gate_in_kernel:
        assert A_log is not None, "use_gate_in_kernel requires A_log"
    if dt_bias is not None:
        assert dt_bias.numel() == HV * K, "dt_bias must be [HV, K]"

    if cu_seqlens is not None:
        avg_T = max(1, T // max(1, N))
    else:
        avg_T = T
    if SK is None:
        SK = 16 if avg_T > 1 and K % 16 == 0 and BV * 16 >= 64 else math.gcd(32, K)
        if num_warps is None and avg_T > 1:
            num_warps = max(1, min(2, BV * SK // 32))
    if num_warps is None:
        num_warps = max(1, min(4, BV * SK // 32))
    assert V % BV == 0, f"BV={BV} must divide V={V}"
    assert 32 % SK == 0 and K % SK == 0, f"SK={SK} must divide 32 and K={K}"
    assert (BV * SK) % (
        32 * num_warps
    ) == 0, f"BV*SK={BV * SK} must be a multiple of 32*{num_warps}"

    # [V, K] or the reference's default [K, V]; the innermost dim sets the row size
    state_shape = (HV, V, K) if state_v_first else (HV, K, V)
    row = K if state_v_first else V

    is_paged = ssm_state_indices is not None
    if is_paged:
        stride_indices_seq = (
            ssm_state_indices.stride(0) if ssm_state_indices.ndim > 1 else 1
        )
    else:
        stride_indices_seq = 1
        assert num_accepted_tokens is None, "spec decoding requires ssm_state_indices"

    if initial_state is not None:
        assert initial_state.dtype == torch.float32
        assert (
            initial_state.shape[1:] == state_shape
        ), f"state must be [slots, HV, {state_shape[1]}, {state_shape[2]}]"

    if inplace_final_state:
        assert initial_state is not None, "inplace_final_state requires initial_state"
        final_state = initial_state
    elif output_final_state:
        if state_out is not None:
            final_state = state_out
        elif is_paged:
            raise ValueError(
                "paged with inplace_final_state=False requires state_out of shape "
                "[total_T, HV, V, K] -- snapshots are indexed by token, not by slot"
            )
        else:
            final_state = q.new_empty(N, *state_shape, dtype=torch.float32)
    else:
        final_state = None

    if final_state is not None:
        assert final_state.dtype == torch.float32
        assert (
            final_state.shape[1:] == state_shape
        ), "final state layout must match initial_state"
        if is_paged and not inplace_final_state:
            assert final_state.shape[0] >= B * T, "state_out needs one slot per token"

    if out is None:
        out = torch.zeros_like(v)
    else:
        assert out.shape == v.shape, "out must match v"

    def _rows(t):
        if t is None:
            return 1
        st = t.stride()
        assert st[0] % row == 0, "state slot stride must be a whole number of rows"
        assert st[1:] == (
            state_shape[1] * state_shape[2],
            state_shape[2],
            1,
        ), "state slabs must be dense in [HV, ., .]"
        return st[0] // row

    slot_rows_in = _rows(initial_state)
    slot_rows_out = (
        slot_rows_in if final_state is initial_state else _rows(final_state)
    )
    assert num_buffers >= 1, "num_buffers must be >= 1"
    if load_buffers is None:
        load_buffers = 2
    assert load_buffers in (
        1,
        2,
        3,
    ), "load_buffers must be 1 (sync), 2 (register prefetch) or 3 (TDM ring)"

    if _LOG_INFO:
        _LOGGER.info(
            f"KDA_DECODE: B={B} T={T} N={N} H={H} HV={HV} K={K} V={V} BV={BV} "
            f"warps={num_warps} paged={is_paged} gate={use_gate_in_kernel}"
        )

    grid = (triton.cdiv(V, BV) * N * HV,)
    fused_recurrent_kda_packed_decode_kernel[grid](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        g_ptr=g,
        beta_ptr=beta,
        A_log_ptr=A_log,
        dt_bias_ptr=dt_bias,
        o_ptr=out,
        state_ptr=initial_state,
        state_out_ptr=final_state,
        cu_seqlens_ptr=cu_seqlens,
        state_indices_ptr=ssm_state_indices,
        num_accepted_ptr=num_accepted_tokens,
        lower_bound=lower_bound if lower_bound is not None else 0.0,
        T=T,
        stride_indices_seq=stride_indices_seq,
        stride_state_slot_rows=slot_rows_in,
        stride_state_out_slot_rows=slot_rows_out,
        scale=scale,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BV=BV,
        NUM_WARPS=num_warps,
        SK=SK,
        NUM_BUFFERS=num_buffers,
        LOAD_BUFFERS=load_buffers,
        NT_STREAM=nt_stream,
        IS_VARLEN=cu_seqlens is not None,
        IS_CONTINUOUS_BATCHING=is_paged,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_GATE_IN_KERNEL=use_gate_in_kernel,
        HAS_DT_BIAS=dt_bias is not None,
        USE_LOWER_BOUND=lower_bound is not None,
        APPLY_BETA_SIGMOID=use_beta_sigmoid_in_kernel,
        ALLOW_NEG_EIGVAL=allow_neg_eigval,
        INPLACE_FINAL_STATE=inplace_final_state,
        STATE_V_FIRST=state_v_first,
        num_warps=num_warps,
    )
    return out, final_state
