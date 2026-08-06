# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import warnings

import torch
import triton
from packaging.version import Version

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

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
    num_warps: int | None = None,
    num_buffers: int = 2,
    lds_pad: int = 4,
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

    if num_warps is None:
        num_warps = max(1, min(8, BV // 32))
    assert V % BV == 0, f"BV={BV} must divide V={V}"
    assert BV % (32 * num_warps) == 0, f"BV={BV} must be a multiple of 32*{num_warps}"

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
            return 1, 1
        assert (
            t.stride(0) % row == 0
        ), "state slot stride must be a whole number of rows"
        assert t.stride()[1:] == (
            state_shape[1] * state_shape[2],
            state_shape[2],
            1,
        ), "state slabs must be dense in [HV, ., .]"
        sr = t.stride(0) // row
        return sr, t.shape[0] * sr

    slot_rows_in, rows_in = _rows(initial_state)
    slot_rows_out, rows_out = _rows(final_state)
    assert num_buffers >= 1, "num_buffers must be >= 1"

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
        state_rows=rows_in,
        state_out_rows=rows_out,
        scale=scale,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BV=BV,
        NUM_WARPS=num_warps,
        NUM_BUFFERS=num_buffers,
        LDS_PAD=lds_pad,
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
