# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Chunked Kimi Delta Attention prefill, Gluon on gfx1250 and gfx950: prepare, then the walk."""

import torch

from aiter.ops.triton.utils._triton import arch_info

_ARCH = arch_info.get_arch()
if _ARCH == "gfx950":
    from aiter.ops.triton._gluon_kernels.gfx950.attention.chunk_kda import prepare, walk
else:
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.chunk_kda import (
        prepare,
        walk,
    )

CHUNK_SIZE = 64
HEAD_DIM = 128
_INDEX_DTYPES = (torch.int32, torch.int64)


def prepare_chunk_kda_metadata(
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """int32 chunk_indices [NT, 2] and int64 chunk_offsets [N + 1]; syncs to read NT."""
    counts = torch.div(
        (cu_seqlens[1:] - cu_seqlens[:-1]).long() + CHUNK_SIZE - 1,
        CHUNK_SIZE,
        rounding_mode="floor",
    )
    chunk_offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    NT = int(chunk_offsets[-1])
    seq = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device), counts, output_size=NT
    )
    chunk = torch.arange(NT, device=counts.device) - chunk_offsets[seq]
    return torch.stack([seq, chunk], 1).int(), chunk_offsets


def _check_tokens(name: str, x: torch.Tensor, D: int) -> None:
    assert x.ndim == 4 and x.shape[0] == 1, f"{name} must be [1, T, H, {D}]"
    assert x.stride()[2:] == (D, 1), f"{name} must be dense in [H, {D}]"


def _check_index(name: str, x: torch.Tensor, shape: tuple) -> None:
    assert (
        x.shape == shape and x.dtype in _INDEX_DTYPES and x.is_contiguous()
    ), f"{name} must be a contiguous int32 / int64 {list(shape)}"


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
    config: dict | None = None,
) -> dict[str, torch.Tensor]:
    """The walk's workspace in one launch; see chunk_kda."""
    assert _ARCH in (
        "gfx1250",
        "gfx950",
    ), f"chunk kda needs gfx1250 or gfx950, got {_ARCH}"
    _, T, H, K = q.shape
    V = v.shape[-1]
    assert K == HEAD_DIM and V == HEAD_DIM, "chunk kda is specialised to K = V = 128"
    # the prepare's 16-token bands keep 2^-G finite only down to this floor
    assert -5.5 <= lower_bound < 0, "lower_bound must be in [-5.5, 0)"
    for name, x, D in (("q", q, K), ("k", k, K), ("v", v, V), ("g", g, K)):
        _check_tokens(name, x, D)
        assert x.shape[1:3] == (T, H), f"{name} must share [T, H] with q"
        assert x.stride(1) % 8 == 0, f"{name} rows must be 16-byte aligned"
    assert beta.shape == (1, T, H) and beta.stride(2) == 1, "beta must be [1, T, H]"
    assert A_log.numel() == H and A_log.is_contiguous(), "A_log must be [H]"
    assert (
        dt_bias.numel() == H * K and dt_bias.is_contiguous()
    ), "dt_bias must be [H * K]"
    _check_index("cu_seqlens", cu_seqlens, (cu_seqlens.numel(),))
    if scale is None:
        scale = K**-0.5
    if chunk_indices is None:
        chunk_indices, _ = prepare_chunk_kda_metadata(cu_seqlens)
    NT = chunk_indices.shape[0]
    _check_index("chunk_indices", chunk_indices, (NT, 2))

    ws = {
        "qg": q.new_empty(1, T, H, K),
        "w": q.new_empty(1, T, H, K),
        "u": q.new_empty(1, T, H, V),
        "kg_t": q.new_empty(NT, H, K, CHUNK_SIZE),
        "aqk": q.new_empty(1, T, H, CHUNK_SIZE),
        "decay": q.new_empty(NT, H, K, dtype=torch.float32),
    }
    config = prepare._get_config(config)
    prepare.chunk_kda_prepare_kernel[(NT, H)](
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
        NUM_WARPS=config["num_warps"],
        NC=config["nc"],
        num_warps=config["num_warps"],
        waves_per_eu=config["waves_per_eu"],
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
    """Chunk recurrence and output from the chunk_kda_prepare workspace; see chunk_kda."""
    assert _ARCH in (
        "gfx1250",
        "gfx950",
    ), f"chunk kda needs gfx1250 or gfx950, got {_ARCH}"
    _, T, H, K = qg.shape
    V = u.shape[-1]
    N = cu_seqlens.numel() - 1
    NT = kg_t.shape[0]
    paged = state_cache is not None
    fuse_norm = out_gate is not None
    _check_index("cu_seqlens", cu_seqlens, (N + 1,))
    for name, x, shape in (
        ("qg", qg, (1, T, H, K)),
        ("w", w, (1, T, H, K)),
        ("u", u, (1, T, H, V)),
        ("aqk", aqk, (1, T, H, CHUNK_SIZE)),
        ("kg_t", kg_t, (NT, H, K, CHUNK_SIZE)),
        ("decay", decay, (NT, H, K)),
    ):
        assert x.shape == shape and x.is_contiguous(), f"{name} must be dense {shape}"
    assert decay.dtype == torch.float32, "decay must be fp32"
    if scale is None:
        scale = K**-0.5
    if chunk_offsets is None:
        _, chunk_offsets = prepare_chunk_kda_metadata(cu_seqlens)
    _check_index("chunk_offsets", chunk_offsets, (N + 1,))
    if out is None:
        out = qg.new_empty(1, T, H, V)
    _check_tokens("out", out, V)
    assert (
        out.shape == (1, T, H, V) and out.dtype == qg.dtype
    ), f"out must be [1, {T}, {H}, {V}] {qg.dtype}"
    state_shape = (H, V, K)
    if paged:
        assert (
            initial_state is None and not output_final_state
        ), "state_cache replaces initial_state / output_final_state"
        assert (
            state_indices is not None and has_initial_state is not None
        ), "state_cache needs state_indices and has_initial_state"
        assert (
            state_indices.dim() == 1
            and state_indices.numel() == N
            and state_indices.dtype in _INDEX_DTYPES
        ), "state_indices must be int32 / int64 [N]"
        assert has_initial_state.numel() == N, "has_initial_state must be [N]"
        has_initial_state = has_initial_state.contiguous()
        state_in = state_out = state_cache
        num_slots = state_cache.shape[0]
        final_state = None
    else:
        assert (
            initial_state is None or initial_state.shape[0] == N
        ), "initial_state must be [N, H, V, K]"
        state_in = initial_state
        state_out = final_state = (
            qg.new_empty(N, *state_shape, dtype=torch.float32)
            if output_final_state
            else None
        )
        num_slots = N
    for s in (state_in, state_out):
        if s is not None:
            assert (
                s.dtype == torch.float32 and s.shape[1:] == state_shape
            ), f"state must be fp32 [*, {H}, {V}, {K}]"
            assert s.stride()[1:] == (V * K, K, 1), "state must be dense [*, H, V, K]"
    if fuse_norm:
        _check_tokens("out_gate", out_gate, V)
        assert norm_weight.numel() == V and norm_weight.is_contiguous()
        config = {"BV": V, "num_warps": 4, "KS": 1, **(config or {})}

    args = {
        "qg_ptr": qg,
        "w_ptr": w,
        "u_ptr": u,
        "kg_t_ptr": kg_t,
        "aqk_ptr": aqk,
        "decay_ptr": decay,
        "o_ptr": out,
        "state_ptr": state_in,
        "state_out_ptr": state_out,
        "cu_seqlens_ptr": cu_seqlens,
        "chunk_offsets_ptr": chunk_offsets,
        "state_indices_ptr": state_indices,
        "has_initial_state_ptr": has_initial_state,
        "out_gate_ptr": out_gate,
        "norm_weight_ptr": norm_weight,
        "norm_eps": norm_eps,
        "stride_state_n": state_in.stride(0) if state_in is not None else 0,
        "stride_state_out_n": state_out.stride(0) if state_out is not None else 0,
        "num_slots": num_slots,
        "stride_o_token": out.stride(1),
        "stride_og_token": out_gate.stride(1) if fuse_norm else 0,
        "scale": scale,
        "H": H,
        "K": K,
        "V": V,
        "BT": CHUNK_SIZE,
        "IS_PAGED": paged,
        "FUSE_NORM": fuse_norm,
    }
    if _ARCH == "gfx950":
        _walk_gfx950(args, walk._get_config(N, H, NT, config), N)
        return out, final_state

    config = walk._get_config(N, H, config)
    if paged:
        args["state_indices_ptr"] = state_indices.contiguous()
    walk.chunk_kda_walk_kernel[(N * H * (V // config["BV"]),)](
        **args,
        BV=config["BV"],
        NUM_WARPS=config["num_warps"],
        NUM_STAGES=config["num_stages"],
        USE_INITIAL_STATE=state_in is not None,
        STORE_FINAL_STATE=state_out is not None,
        num_warps=config["num_warps"],
        waves_per_eu=config["waves_per_eu"],
    )
    return out, final_state


def _walk_gfx950(args: dict, config: dict, N: int) -> None:
    H, K, V, G = args["H"], args["K"], args["V"], config["G"]
    qg, state_in, state_out = args["qg_ptr"], args["state_ptr"], args["state_out_ptr"]
    paged = args["IS_PAGED"]
    args = {
        **args,
        "stride_indices": args["state_indices_ptr"].stride(0) if paged else 0,
        "n_seq": N,
    }

    def launch(PASS, bufs, use_init, store_final):
        s = "_pass1" if PASS == 1 else ""
        bv, nw = config["BV" + s], config["num_warps" + s]
        grid = N * H * G * (V // bv)
        if PASS == 1:
            grid = N * H * (V // bv + (G - 2) * ((V + K) // bv))
        walk.chunk_kda_walk_kernel[(grid,)](
            **args,
            bg_ptr=bufs[0],
            mg_ptr=bufs[1],
            sin_ptr=bufs[2],
            n_groups=G,
            BV=bv,
            NUM_WARPS=nw,
            KS=config["KS" + s],
            NUM_STAGES=config["num_stages" + s],
            PASS=PASS,
            USE_INITIAL_STATE=use_init,
            STORE_FINAL_STATE=store_final,
            num_warps=nw,
            waves_per_eu=config["waves_per_eu" + s],
        )

    if G == 1:
        launch(0, (qg,) * 3, state_in is not None, state_out is not None)
        return

    bufs = (
        qg.new_empty(G, N, H, V, K, dtype=torch.float32),  # B_g
        qg.new_empty(G, N, H, K, K),  # M_g^T
        qg.new_empty(G, N, H, V, K, dtype=torch.float32),  # entry states
    )
    launch(1, bufs, state_in is not None, False)
    if G > 2:
        grid = N * H * (V // config["BV_scan"])
        walk.chunk_kda_scan_kernel[(grid,)](
            bg_ptr=bufs[0],
            mg_ptr=bufs[1],
            sin_ptr=bufs[2],
            n_seq=N,
            n_groups=G,
            H=H,
            K=K,
            V=V,
            BV=config["BV_scan"],
            NUM_WARPS=config["num_warps_scan"],
            NUM_STAGES=config["num_stages_scan"],
            num_warps=config["num_warps_scan"],
            waves_per_eu=config["waves_per_eu_scan"],
        )
    launch(2, bufs, False, state_out is not None)


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
    """Chunked Kimi Delta Attention prefill from raw projections (Gluon, gfx1250 / gfx950).

    Two launches. ``chunk_kda_prepare`` fuses the q/k l2norm, the gate and its cumsum, the
    intra-chunk products and the (I + L)^-1 solve into a workspace: qg, w, u [1, T, H, 128],
    aqk [1, T, H, 64], kg_t [NT, H, 128, 64], fp32 decay [NT, H, 128]. ``chunk_kda_walk``
    then runs the recurrence on the fp32 V-first state S [V, K], per chunk
    v = u - w S^T, o = scale qg S^T + aqk v, S = S diag(decay) + v^T kg.

    Args:
        q, k, v: [1, T, H, 128] raw projections; token rows may be strided (16 B aligned).
        g: [1, T, H, 128] raw gate projection, before the activation.
        beta: [1, T, H] raw beta projection, before the sigmoid.
        A_log: [H] gate parameter. dt_bias: [H * 128] gate bias.
        lower_bound: gate floor in [-5.5, 0); g = lower_bound * sigmoid(exp(A_log) (g + dt_bias)).
        cu_seqlens: int32 / int64 [N + 1] varlen offsets.
        chunk_indices, chunk_offsets: int [NT, 2] (sequence, chunk) and [N + 1] first chunk
            per sequence; ``prepare_chunk_kda_metadata`` (a host sync) if None.
        scale: q scale, K**-0.5 by default.
        out: [1, T, H, 128] destination, may alias the dead v; allocated if None.
        initial_state: fp32 [N, H, V, K] start state, zeros if None.
        output_final_state: return a fresh fp32 [N, H, V, K] final state.
        state_cache: fp32 [slots, H, V, K] paged state, any slot stride, read and written in
            place at ``state_indices``; replaces initial_state / output_final_state.
        state_indices: int32 / int64 [N] cache row per sequence, any stride. Valid rows must
            be distinct; a row outside [0, slots) (PAD_SLOT_ID) is never read or written:
            that sequence starts from zeros and its final state is dropped.
        has_initial_state: bool [N]; False starts that sequence from zeros.
        out_gate, norm_weight, norm_eps: fuse o = rmsnorm(o) * norm_weight * sigmoid(out_gate).
        config: overrides of the tuned walk config (the kernels' DEFAULT.json).

    Returns:
        (o, final_state); final_state is None when the state lives in state_cache.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if chunk_indices is None or chunk_offsets is None:
        ci, co = prepare_chunk_kda_metadata(cu_seqlens)
        chunk_indices = ci if chunk_indices is None else chunk_indices
        chunk_offsets = co if chunk_offsets is None else chunk_offsets
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
