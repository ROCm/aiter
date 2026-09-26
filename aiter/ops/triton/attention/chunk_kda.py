# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import logging

import torch

from aiter.ops.triton._triton_kernels.gated_delta_rule.utils.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.config_utils import (
    AITER_TRITON_CONFIGS_PATH,
    load_config_json,
)
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
_LOG_INFO = _LOGGER._logger.isEnabledFor(logging.INFO)

_ARCH = arch_info.get_arch()

CHUNK_SIZE = 64
HEAD_DIM = 128

chunk_kda_prepare_kernel = chunk_kda_walk_kernel = None
if _ARCH == "gfx1250":
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.chunk_kda.prepare import (
        chunk_kda_prepare_kernel,
    )
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.chunk_kda.walk import (
        chunk_kda_walk_kernel,
    )


def get_chunk_kda_config(num_seqs: int, H: int, overrides: dict | None = None) -> dict:
    tuned = load_config_json(
        f"{AITER_TRITON_CONFIGS_PATH}/{_ARCH}-CHUNK_KDA-DEFAULT.json"
    )
    # the walk is serial in chunks: when (seq, head) pairs cannot fill the GPU at
    # BV=64, a narrower BV buys programs
    few = num_seqs * H * (HEAD_DIM // 64) < get_num_sms()
    config = dict(tuned["walk_few_seq_heads" if few else "walk_default"])
    config.update(overrides or {})
    return config


def _check_tokens(name: str, x: torch.Tensor, D: int) -> None:
    assert x.ndim == 4 and x.shape[0] == 1, f"{name} must be [1, T, H, {D}]"
    assert x.stride()[2:] == (D, 1), f"{name} must be dense in [H, {D}]"


def chunk_kda_prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor | None = None,
    scale: float | None = None,
) -> dict[str, torch.Tensor]:
    """Per-chunk operands of the chunked KDA walk, one launch (gfx1250 Gluon).

    Fuses the q/k l2norm, the lower-bounded gate, its cumsum, the intra-chunk
    L / Aqk products, the (I + L)^-1 solve and w/u. The workspace matches the
    ROCm HIP prologue: qg, w, u token-major [1, T, H, 128]; aqk token-major
    [1, T, H, 64] (scale folded in, qg unscaled); kg_t chunk-major
    [NT, H, 128, 64]; decay fp32 [NT, H, 128].

    Args:
        q, k, v: [1, T, H, 128] raw projections; may be strided token-major views.
        g: [1, T, H, 128] raw gate projection, before the activation.
        beta: [1, T, H] raw beta projection, before the sigmoid.
        A_log: [H] gate parameter.
        dt_bias: [H * 128] gate bias.
        lower_bound: gate floor; g = lower_bound * sigmoid(exp(A_log) (g + dt_bias)).
        cu_seqlens: [N + 1] varlen offsets.
        chunk_indices: [NT, 2] (sequence, chunk) pairs for 64-token chunks.
        scale: q scale, K**-0.5 by default.
    """
    if chunk_kda_prepare_kernel is None:
        raise RuntimeError(f"chunk kda gluon requires gfx1250 (found {_ARCH})")
    _, T, H, K = q.shape
    V = v.shape[-1]
    assert K == HEAD_DIM and V == HEAD_DIM, "chunk kda is specialised to K = V = 128"
    # 16-token bands keep 2^(pivot - G) below 2^116 only while each gate step is >= lb log2(e)
    assert -5.5 <= lower_bound < 0, "lower_bound must be in [-5.5, 0)"
    for name, x, D in (("q", q, K), ("k", k, K), ("v", v, V), ("g", g, K)):
        _check_tokens(name, x, D)
        assert x.shape[1:3] == (T, H), f"{name} must share [T, H] with q"
        assert x.stride(1) % 8 == 0, f"{name} rows must be 16-byte aligned for TDM"
    assert beta.shape == (1, T, H) and beta.stride(2) == 1, "beta must be [1, T, H]"
    assert A_log.numel() == H and A_log.is_contiguous(), "A_log must be [H]"
    assert (
        dt_bias.numel() == H * K and dt_bias.is_contiguous()
    ), "dt_bias must be [H * K]"
    assert cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous"
    if scale is None:
        scale = K**-0.5
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    assert chunk_indices.is_contiguous(), "chunk_indices must be contiguous"
    NT = chunk_indices.shape[0]

    ws = dict(
        qg=q.new_empty(1, T, H, K),
        w=q.new_empty(1, T, H, K),
        u=q.new_empty(1, T, H, V),
        kg_t=q.new_empty(NT, H, K, CHUNK_SIZE),
        aqk=q.new_empty(1, T, H, CHUNK_SIZE),
        decay=q.new_empty(NT, H, K, dtype=torch.float32),
    )
    if _LOG_INFO:
        _LOGGER.info(f"CHUNK_KDA_PREPARE: T={T} NT={NT} H={H}")
    chunk_kda_prepare_kernel[(NT, H)](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        g_ptr=g,
        beta_ptr=beta,
        A_log_ptr=A_log,
        dt_bias_ptr=dt_bias,
        qg_ptr=ws["qg"],
        w_ptr=ws["w"],
        u_ptr=ws["u"],
        kg_t_ptr=ws["kg_t"],
        aqk_ptr=ws["aqk"],
        decay_ptr=ws["decay"],
        cu_seqlens_ptr=cu_seqlens,
        chunk_indices_ptr=chunk_indices,
        lower_bound=lower_bound,
        stride_q_token=q.stride(1),
        stride_k_token=k.stride(1),
        stride_v_token=v.stride(1),
        stride_g_token=g.stride(1),
        stride_beta_token=beta.stride(1),
        scale=scale,
        H=H,
        K=K,
        V=V,
        BT=CHUNK_SIZE,
        num_warps=4,
    )
    return ws


def chunk_kda_walk(
    qg: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    kg_t: torch.Tensor,
    aqk: torch.Tensor,
    decay: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor | None = None,
    scale: float | None = None,
    out: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_cache: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    out_gate: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-5,
    config: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunk recurrence and output from the ``chunk_kda_prepare`` workspace, one launch.

    Per chunk, with the fp32 V-first state S [V, K]:
    v_new = u - w S^T, o = scale qg S^T + aqk v_new, S = S diag(decay) + v_new^T kg.

    Args:
        out: [1, T, H, 128] destination, may alias the dead v; allocated if None.
        initial_state: fp32 [N, H, V, K] per-sequence start state, zeros if None.
        output_final_state: return a fresh fp32 [N, H, V, K] final state.
        state_cache: fp32 [slots, H, V, K] paged state, read and written in
            place at ``state_indices``; replaces initial_state / output_final_state.
        state_indices: int32 [N] cache row per sequence.
        has_initial_state: bool [N]; False starts that sequence from zeros.
        out_gate, norm_weight: fuse o = rmsnorm(o) * norm_weight * sigmoid(out_gate).
        config: overrides for BV, num_warps, num_stages, waves_per_eu.

    Returns:
        (out, final_state); final_state is None when the state lives in state_cache.
    """
    if chunk_kda_walk_kernel is None:
        raise RuntimeError(f"chunk kda gluon requires gfx1250 (found {_ARCH})")
    _, T, H, K = qg.shape
    V = u.shape[-1]
    N = cu_seqlens.numel() - 1
    paged = state_cache is not None
    fuse_norm = out_gate is not None
    if scale is None:
        scale = K**-0.5
    if chunk_offsets is None:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE)
    if out is None:
        out = qg.new_empty(1, T, H, V)
    _check_tokens("out", out, V)
    state_shape = (H, V, K)
    if paged:
        assert (
            initial_state is None and not output_final_state
        ), "state_cache replaces initial_state / output_final_state"
        assert state_indices is not None and has_initial_state is not None
        assert state_indices.numel() == N and state_indices.is_contiguous()
        assert has_initial_state.numel() == N and has_initial_state.is_contiguous()
        state_in = state_out = state_cache
        final_state = None
    else:
        state_in = initial_state
        state_out = final_state = (
            qg.new_empty(N, *state_shape, dtype=torch.float32)
            if output_final_state
            else None
        )
    for s in (state_in, state_out):
        if s is not None:
            assert s.dtype == torch.float32 and s.shape[1:] == state_shape
            assert s.stride()[1:] == (V * K, K, 1), "state must be dense [*, H, V, K]"
    if fuse_norm:
        _check_tokens("out_gate", out_gate, V)
        assert norm_weight.numel() == V and norm_weight.is_contiguous()

    config = get_chunk_kda_config(N, H, config)
    BV = config["BV"]
    num_warps = config["num_warps"]
    assert (
        V % BV == 0 and BV % (16 * num_warps) == 0
    ), f"illegal BV={BV}, warps={num_warps}"
    assert not fuse_norm or BV == V, "the fused norm needs BV == V"
    opts = (
        {"waves_per_eu": config["waves_per_eu"]} if config.get("waves_per_eu") else {}
    )

    if _LOG_INFO:
        _LOGGER.info(
            f"CHUNK_KDA_WALK: T={T} N={N} H={H} BV={BV} warps={num_warps} "
            f"paged={paged} fuse_norm={fuse_norm}"
        )
    chunk_kda_walk_kernel[(N * H * (V // BV),)](
        qg_ptr=qg,
        w_ptr=w,
        u_ptr=u,
        kg_t_ptr=kg_t,
        aqk_ptr=aqk,
        decay_ptr=decay,
        o_ptr=out,
        state_ptr=state_in,
        state_out_ptr=state_out,
        cu_seqlens_ptr=cu_seqlens,
        chunk_offsets_ptr=chunk_offsets,
        state_indices_ptr=state_indices,
        has_initial_state_ptr=has_initial_state,
        out_gate_ptr=out_gate,
        norm_weight_ptr=norm_weight,
        norm_eps=norm_eps,
        stride_o_token=out.stride(1),
        stride_og_token=out_gate.stride(1) if fuse_norm else 0,
        scale=scale,
        H=H,
        K=K,
        V=V,
        BT=CHUNK_SIZE,
        BV=BV,
        NUM_WARPS=num_warps,
        NUM_STAGES=config.get("num_stages", 2),
        IS_PAGED=paged,
        USE_INITIAL_STATE=state_in is not None,
        STORE_FINAL_STATE=state_out is not None,
        FUSE_NORM=fuse_norm,
        num_warps=num_warps,
        **opts,
    )
    return out, final_state


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    scale: float | None = None,
    out: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_cache: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    out_gate: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-5,
    config: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Chunked Kimi Delta Attention prefill from raw projections, gfx1250 Gluon.

    Two launches, ``chunk_kda_prepare`` then ``chunk_kda_walk``; see both for
    the arguments. Returns (o, final_state) like ``chunk_kda_walk``.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    ws = chunk_kda_prepare(
        q, k, v, g, beta, A_log, dt_bias, lower_bound, cu_seqlens, chunk_indices, scale
    )
    return chunk_kda_walk(
        **ws,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        out=out,
        initial_state=initial_state,
        output_final_state=output_final_state,
        state_cache=state_cache,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        out_gate=out_gate,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        config=config,
    )
