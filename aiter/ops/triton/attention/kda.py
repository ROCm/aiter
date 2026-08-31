# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import logging
import math

import torch
import triton
from packaging.version import Version

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH, load_config_json
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
    cache_state_updates: bool = False,
    pad_slot_guard: bool = False,
    config: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fused recurrent Kimi Delta Attention (KDA), gfx1250 Gluon decode path.
    Args:
        q: [B, T, H, K] queries
        k: [B, T, H, K] keys
        v: [B, T, HV, V] values; HV must be a multiple of H (q/k heads are
            broadcast across HV // H value heads).
        g: [B, T, HV, K] log-space decay gates
        beta: [B, T, HV] scalar or [B, T, HV, V] headwise write strengths.
        A_log: [HV] gate parameter
        dt_bias: [HV, K] optional bias added to g inside the gate chain.
        initial_state: fp32 state slabs, [slots, HV, V, K] when
            ``state_v_first`` (default) else [slots, HV, K, V]
        scale: q scale factor; defaults to K**-0.5.
        output_final_state: return the post-recurrence state
        inplace_final_state: update ``initial_state``'s slabs in place, requires ``initial_state``.
        state_v_first: state slab layout = [V, K]
        cu_seqlens: [N + 1] varlen offsets; requires B == 1 with all tokens
            flattened into T.
        ssm_state_indices: [N, T] (or flat) per-token state slot table
        num_accepted_tokens: [N] accepted-token counts for speculative
        use_qk_l2norm_in_kernel: l2-normalize q and k rows in-kernel.
        use_gate_in_kernel: fuse the KDA gate chain
            g = -exp(A_log) * softplus(g + dt_bias), or its lower-bounded
            variant when ``lower_bound`` is set, into the kernel.
        use_beta_sigmoid_in_kernel: apply sigmoid to beta in-kernel.
        allow_neg_eigval: with the in-kernel sigmoid, scale beta by 2 so
            state eigenvalues may go negative.
        lower_bound: bound for the lower-bounded gate chain
        out: optional output buffer, v's shape and dtype.
        state_out: optional final-state destination when not in-place
        cache_state_updates: paged + in-place + v-first only: store the
            full slab only for a sequence's first token, then compact
            (a, k, err) records per token for replay after verification.
        pad_slot_guard: paged only: state slots <= 0 mark padded sequences
        config: tuning dict (BV, SK, num_warps, num_buffers, use_tdm_store,
            use_tdm_load, use_tdm_fused_load); None resolves a tuned bucket
            from ``configs/<arch>-KDA_DECODE-DEFAULT.json``, and omitted
            keys fall back to defaults or a shape-based derivation.

    Returns:
        (o, final_state): o is [B, T, HV, V] in v's dtype. final_state is
        the fp32 state tensor (``initial_state`` itself when in-place, else
        ``state_out`` or a fresh [N, HV, ., .] tensor), or None when
        neither ``output_final_state`` nor ``inplace_final_state`` is set.
    """

    if fused_recurrent_kda_packed_decode_kernel is None:
        raise RuntimeError(
            f"kda gluon decode requires triton>=3.6.0 on gfx1250 "
            f"(found triton {triton.__version__} on {_ARCH})"
        )
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError(
            "allow_neg_eigval=True requires use_beta_sigmoid_in_kernel=True"
        )

    if state_v_first is None:
        state_v_first = True

    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    if scale is None:
        scale = K**-0.5

    assert k.shape == q.shape, "q and k must match"
    assert v.shape[:2] == q.shape[:2], "v must share [B, T] with q"
    assert g.shape == (B, T, HV, K), "g must be [B, T, HV, K]"
    assert beta.shape in (
        (B, T, HV),
        (B, T, HV, V),
    ), "beta must be [B, T, HV] or [B, T, HV, V]"
    assert HV % H == 0, "HV must be divisible by H"
    assert q.stride()[2:] == (K, 1), "q must be dense in [H, K]"
    assert k.stride()[2:] == (K, 1), "k must be dense in [H, K]"
    assert v.stride()[2:] == (V, 1), "v must be dense in [HV, V]"
    assert g.stride()[2:] == (K, 1), "g must be dense in [HV, K]"
    if beta.ndim == v.ndim:
        assert beta.stride()[2:] == (V, 1), "headwise beta must be dense in [HV, V]"
    else:
        assert beta.stride(2) == 1, "beta heads must be contiguous"
    if B > 1:
        for x in (q, k, v, g, beta):
            assert x.stride(0) == T * x.stride(1), "batches must be token-major"
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
        assert A_log.numel() == HV, "A_log must be [HV]"
    if dt_bias is not None:
        assert dt_bias.numel() == HV * K, "dt_bias must be [HV, K]"
    assert A_log is None or A_log.is_contiguous(), "A_log must be contiguous"
    assert dt_bias is None or dt_bias.is_contiguous(), "dt_bias must be contiguous"
    assert (
        cu_seqlens is None or cu_seqlens.is_contiguous()
    ), "cu_seqlens must be contiguous"

    if cu_seqlens is not None:
        avg_T = max(1, T // max(1, N))
    else:
        avg_T = T
    if config is None:
        # Tuned buckets from the config file; keys a bucket omits fall back to
        # the legacy derivation below.
        tuned = load_config_json(
            f"{AITER_TRITON_CONFIGS_PATH}/{_ARCH}-KDA_DECODE-DEFAULT.json"
        )
        num_seq_heads = N * HV
        config = tuned["default"]
        if K % 32 == 0 and V % 32 == 0:
            if avg_T > 1:
                if num_seq_heads >= 3072 and V % 128 == 0:
                    config = tuned["t_gt1_seq_heads_geq_3072"]
                elif num_seq_heads >= 384:
                    config = tuned["t_gt1_seq_heads_geq_384"]
            elif 256 <= num_seq_heads <= 512:
                config = tuned["t1_seq_heads_256_to_512"]
    BV = config.get("BV", 32)
    SK = config.get("SK")
    num_warps = config.get("num_warps")
    num_buffers = config.get("num_buffers", 2)
    use_tdm_store = config.get("use_tdm_store", False)
    use_tdm_load = config.get("use_tdm_load", False)
    use_tdm_fused_load = config.get("use_tdm_fused_load", False)
    if SK is None:
        if avg_T > 1 and K % 16 == 0 and BV * 16 >= 64:
            SK = 8 if (V // BV) * N * HV <= 8192 else 16
        else:
            SK = math.gcd(32, K)
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
        assert ssm_state_indices.stride(-1) == 1, "state index rows must be dense"
        assert (
            num_accepted_tokens is None or num_accepted_tokens.is_contiguous()
        ), "num_accepted_tokens must be contiguous"
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
        out = torch.empty_like(v)
    else:
        assert out.shape == v.shape, "out must match v"
    assert out.stride()[2:] == (V, 1), "out must be dense in [HV, V]"
    if B > 1:
        assert out.stride(0) == T * out.stride(1), "out batches must be token-major"

    def _rows(t):
        if t is None:
            return 1, 1
        st = t.stride()
        assert st[0] % row == 0, "state slot stride must be a whole number of rows"
        assert st[1:] == (
            state_shape[1] * state_shape[2],
            state_shape[2],
            1,
        ), "state slabs must be dense in [HV, ., .]"
        sr = st[0] // row
        return sr, t.shape[0] * sr

    slot_rows_in, rows_in = _rows(initial_state)
    slot_rows_out, rows_out = (
        (slot_rows_in, rows_in) if final_state is initial_state else _rows(final_state)
    )
    assert num_buffers in (1, 2), "num_buffers: 1 sync, 2 register prefetch"
    if cache_state_updates:
        assert is_paged, "cache_state_updates requires ssm_state_indices"
        assert inplace_final_state, "cache_state_updates requires inplace_final_state"
        assert not use_tdm_store, "cache_state_updates is incompatible with tdm_store"
    if pad_slot_guard:
        assert is_paged, "pad_slot_guard requires ssm_state_indices"

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
        stride_q_token=q.stride(1),
        stride_k_token=k.stride(1),
        stride_v_token=v.stride(1),
        stride_g_token=g.stride(1),
        stride_beta_token=beta.stride(1),
        stride_o_token=out.stride(1),
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
        SK=SK,
        NUM_BUFFERS=num_buffers,
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
        USE_TDM_STORE=use_tdm_store,
        USE_TDM_LOAD=use_tdm_load,
        CACHE_STATE_UPDATES=cache_state_updates,
        PAD_SLOT_GUARD=pad_slot_guard,
        USE_TDM_FUSED_LOAD=use_tdm_fused_load,
        num_warps=num_warps,
    )
    return out, final_state
