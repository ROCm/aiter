# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Batch-invariant MXFP4 MoE forward (SiTU, top-16), Gluon kernels (gfx950)."""

import functools
from collections.abc import Callable

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx950.moe.batch_invariant.moe_mxfp4_bi import (
    _moe_mxfp4_bi_combine_gluon_kernel,
    _moe_mxfp4_bi_gemm_gluon_kernel,
    _moe_mxfp4_bi_gu_situ_async_gluon_kernel,
    _moe_mxfp4_bi_gu_situ_gluon_kernel,
    _moe_mxfp4_bi_route_count_gluon_kernel,
    _moe_mxfp4_bi_route_gluon_kernel,
    _moe_mxfp4_bi_route_scan_gluon_kernel,
    _moe_mxfp4_bi_route_scatter_gluon_kernel,
    _moe_mxfp4_bi_tree8_rows_gluon_kernel,
    _moe_mxfp4_bi_tree16_plane_gluon_kernel,
    _moe_mxfp4_bi_w2_tree_async_gluon_kernel,
    _moe_mxfp4_bi_w2_tree_gluon_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.moe_config_utils import get_moe_dispatch

__all__ = ["BI_CONTRACT_VERSION", "moe_mxfp4_bi_forward"]

# Bump when a change can change the output bits; callers compare it at startup.
BI_CONTRACT_VERSION = 1

N_CHUNKS = 8  # the intermediate dimension is always summed in 8 fixed chunks
TOP_K = 16
# Tokens per collective call (leaf_reduce, chunk_gather): part of the contract.
TOKEN_BLOCK_COLLECTIVE = 1024
_MIN_BLOCK_K = 64  # smallest legal K step: H and N / 8 are multiples of 64
_LAUNCH = ("num_warps", "num_stages")


@functools.cache
def _config(name: str) -> dict:
    """Pinned launch values of one kernel (constexpr names upper case)."""
    cfg = get_moe_dispatch("MOE-MXFP4-BI", get_arch(), "gluon").get(name)
    if cfg is None:
        raise NotImplementedError(
            f"moe_mxfp4_bi: no pinned config '{name}' for this arch"
        )
    return {k if k in _LAUNCH else k.upper(): v for k, v in cfg.items()}


def _block_m(num_tokens: int, n_experts_global: int) -> int:
    """16-row tiles while the experts get few rows each (weight bandwidth), else 64-row
    tiles (fewer fp4 upcasts per FLOP). The bits do not depend on the choice."""
    c = _config("block_m")
    rows = num_tokens * TOP_K / n_experts_global
    return c["BLOCK_M_SMALL"] if rows <= c["ROWS_SMALL"] else c["BLOCK_M_LARGE"]


def _route_fused(topk_ids, expert_map, n_experts, block_m):
    """Stable counting sort of the slots by local expert: (tok, rows_tok, expt_offs,
    tile_offs, max_tiles, ord)."""
    T = topk_ids.shape[0]
    S = T * TOP_K
    dev = topk_ids.device
    rc = _config("route")
    NB = triton.next_power_of_2(n_experts + 1)
    ids = topk_ids.contiguous()
    tok = torch.empty(S, dtype=torch.int32, device=dev)
    ord_ = torch.empty(S, dtype=torch.int32, device=dev)
    rows_tok = torch.empty((T, TOP_K), dtype=torch.int32, device=dev)
    expt_offs = torch.empty(n_experts + 1, dtype=torch.int32, device=dev)
    tile_offs = torch.empty(n_experts + 1, dtype=torch.int32, device=dev)
    emap = expert_map if expert_map is not None else ids
    has_map = expert_map is not None
    n_ids = expert_map.numel() if has_map else n_experts
    nw = rc["num_warps"]
    if S <= rc["MAX_SLOTS_ONE_PROGRAM"]:
        _moe_mxfp4_bi_route_gluon_kernel[(1,)](
            ids,
            emap,
            tok,
            rows_tok,
            ord_,
            expt_offs,
            tile_offs,
            S,
            n_experts,
            n_ids,
            TOP_K=TOP_K,
            HAS_MAP=has_map,
            BLOCK_M=block_m,
            NB=NB,
            BS=rc["BS"],
            num_warps=nw,
        )
    else:
        blk = rc["BLK"]
        assert blk % rc["BS"] == 0
        nblk = triton.cdiv(S, blk)
        cnt = torch.empty((nblk, NB), dtype=torch.int32, device=dev)
        base = torch.empty((nblk, NB), dtype=torch.int32, device=dev)
        _moe_mxfp4_bi_route_count_gluon_kernel[(nblk,)](
            ids,
            emap,
            cnt,
            S,
            n_experts,
            n_ids,
            HAS_MAP=has_map,
            NB=NB,
            BS=rc["BS"],
            BLK=blk,
            num_warps=nw,
        )
        _moe_mxfp4_bi_route_scan_gluon_kernel[(1,)](
            cnt,
            base,
            expt_offs,
            tile_offs,
            nblk,
            n_experts,
            BLOCK_M=block_m,
            NB=NB,
            num_warps=nw,
        )
        _moe_mxfp4_bi_route_scatter_gluon_kernel[(nblk,)](
            ids,
            emap,
            base,
            tok,
            rows_tok,
            ord_,
            S,
            n_experts,
            n_ids,
            TOP_K=TOP_K,
            HAS_MAP=has_map,
            NB=NB,
            BS=rc["BS"],
            BLK=blk,
            num_warps=nw,
        )
    max_tiles = triton.cdiv(S, block_m) + min(n_experts, S)
    return tok, rows_tok, expt_offs, tile_offs, max_tiles, ord_


def _f32(v: float) -> float:
    return float(torch.tensor(v, dtype=torch.float32))


def _tree8_rows(P, nrows, dtype=torch.float32):
    """[R, N] in dtype: the fixed pair tree of the 8 FP32 chunk planes P [8, R, N] for
    rows < nrows."""
    C, R, N = P.shape
    assert C == N_CHUNKS and P.stride(2) == 1
    out = torch.empty((R, N), dtype=dtype, device=P.device)
    if R:
        c = _config("tree8")
        _moe_mxfp4_bi_tree8_rows_gluon_kernel[
            (triton.cdiv(R, c["BR"]), triton.cdiv(N, c["BN"]))
        ](
            P,
            out,
            nrows,
            N,
            P.stride(0),
            P.stride(1),
            out.stride(0),
            BR=c["BR"],
            BN=c["BN"],
            num_warps=c["num_warps"],
        )
    return out


def _grouped_gemm(a, a_rows, w, w_scale, route, n_out, chunks, block_m):
    """FP32 [chunks, R, n_out]: plane c = a[a_rows, chunk c] @ deq(w[e, :, chunk c])^T
    (tensor-parallel W2)."""
    _, _, _, expt_offs, tile_offs, max_tiles = route
    R = a_rows.numel()
    Kc = w.shape[2] * 2 // chunks
    out = torch.empty((chunks, R, n_out), dtype=torch.float32, device=a.device)
    if R == 0:
        return out
    c = _config("gemm")
    bk = c["BLOCK_K"] if Kc % c["BLOCK_K"] == 0 else _MIN_BLOCK_K
    _moe_mxfp4_bi_gemm_gluon_kernel[
        (max_tiles, triton.cdiv(n_out, c["BLOCK_N"]), chunks)
    ](
        a,
        a_rows,
        w.view(torch.uint8),
        w_scale.view(torch.uint8),
        out,
        expt_offs,
        tile_offs,
        w.shape[0],
        n_out,
        Kc,
        a.stride(0),
        w.stride(0),
        w.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        out.stride(0),
        out.stride(1),
        Kc,
        Kc // 2,
        Kc // 32,
        BLOCK_M=block_m,
        BLOCK_N=c["BLOCK_N"],
        BLOCK_K=bk,
        num_warps=c["num_warps"],
        enable_fp_fusion=False,
    )
    return out


def _gu_situ(x, tok, w13, w13_scale, route, beta, linear_beta, block_m):
    """bf16 [R, N]: SiTU of each slot row's gate / up GEMM."""
    _, _, _, expt_offs, tile_offs, max_tiles = route
    R = tok.numel()
    N = w13.shape[1] // 2
    H = x.shape[1]
    out = torch.empty((R, N), dtype=torch.bfloat16, device=x.device)
    if R == 0:
        return out
    c = _config(f"gu_bm{block_m}")
    bk = c["BLOCK_K"] if H % c["BLOCK_K"] == 0 else _MIN_BLOCK_K
    inv = lambda b: _f32(_f32(1.0) / _f32(b))
    has_l = linear_beta is not None
    # Async copies need 4 scale bytes per row (BLOCK_K 128) and 16-byte rows.
    aligned = _aligned(w13, w13_scale)
    nb = c.get("NUM_BUFFERS", 1) if bk % 128 == 0 and aligned else 1
    pipe = {"NUM_BUFFERS": nb} if nb > 1 else {}
    if nb > 1:
        pipe.update(
            {
                k: c[k]
                for k in (
                    "N_FIRST",
                    "MMA_K",
                    "W_SWZ",
                )
                if k in c
            }
        )
    kernel = (
        _moe_mxfp4_bi_gu_situ_async_gluon_kernel
        if nb > 1
        else _moe_mxfp4_bi_gu_situ_gluon_kernel
    )
    grid = (max_tiles, triton.cdiv(N, c["BLOCK_N"]))
    kernel[grid[::-1] if pipe.get("N_FIRST") else grid](
        x,
        tok,
        w13.view(torch.uint8),
        w13_scale.view(torch.uint8),
        out,
        expt_offs,
        tile_offs,
        w13.shape[0],
        N,
        H,
        x.stride(0),
        w13.stride(0),
        w13.stride(1),
        w13_scale.stride(0),
        w13_scale.stride(1),
        out.stride(0),
        inv(beta),
        _f32(beta),
        inv(linear_beta) if has_l else 1.0,
        _f32(linear_beta) if has_l else 1.0,
        HAS_L=has_l,
        BLOCK_M=block_m,
        BLOCK_N=c["BLOCK_N"],
        BLOCK_K=bk,
        num_warps=c["num_warps"],
        enable_fp_fusion=False,
        **pipe,
    )
    return out


def _aligned(*ts):
    """Base and rows on 16 bytes (the async copies)."""
    return all(
        t.data_ptr() % 16 == 0 and t.stride(0) % 16 == 0 and t.stride(1) % 16 == 0
        for t in ts
    )


def _w2_plan(n_global, block_m, w2, w2_scale):
    """(config, BLOCK_K, async) of the W2 kernel. The kernel without async copies steps
    whole BLOCK_K tiles through each chunk, so its BLOCK_K must divide the chunk."""
    c = _config(f"w2_bm{block_m}")
    chunk = n_global // N_CHUNKS
    bk, mk = c["BLOCK_K"], c.get("MMA_K", c["BLOCK_K"])
    if (
        c.get("NUM_BUFFERS", 1) > 1
        and bk % 128 == 0
        and chunk % mk == 0
        and N_CHUNKS * chunk % bk == 0
        and _aligned(w2, w2_scale)
    ):
        return dict(c), bk, True
    c = {k: v for k, v in c.items() if k != "MMA_K"}
    return c, bk if chunk % bk == 0 else _MIN_BLOCK_K, False


def _w2_async(n_global, block_m, w2, w2_scale):
    """True when _w2 takes the async W2 kernel for this chunk size and weight layout."""
    return _w2_plan(n_global, block_m, w2, w2_scale)[2]


def _tree16_blocks(slots, topk_ids, leaf_reduce, leaf_block, planes=None):
    """bf16 [T, H]: leaf_reduce per leaf_block tokens, then each token's leaf tree;
    planes = (ws, rank): one block of column planes, the tree fills this rank's columns.
    """
    T = topk_ids.shape[0]
    c = _config("combine")
    if planes is not None:
        ws, rank = planes
        hp = slots.shape[2]
        assert T <= leaf_block
        out = torch.empty((T, ws * hp), dtype=torch.bfloat16, device=slots.device)
        red = leaf_reduce(slots).contiguous()
        if T > 0:
            _moe_mxfp4_bi_tree16_plane_gluon_kernel[(T, triton.cdiv(ws * hp, c["BH"]))](
                red,
                topk_ids.contiguous(),
                out,
                ws * hp,
                hp,
                rank * hp,
                out.stride(0),
                BH=c["BH"],
                num_warps=c["num_warps"],
            )
        return out
    H = slots.shape[1]
    out = torch.empty((T, H), dtype=torch.bfloat16, device=slots.device)
    if T == 0:  # the collective is still called once (same calls on every rank)
        leaf_reduce(slots)
    for i in range(0, T, leaf_block):
        j = min(T, i + leaf_block)
        red = leaf_reduce(slots[i * TOP_K : j * TOP_K])
        # the plane kernel with one plane of all H columns
        _moe_mxfp4_bi_tree16_plane_gluon_kernel[(j - i, triton.cdiv(H, c["BH"]))](
            red,
            topk_ids[i:j].contiguous(),
            out[i:j],
            H,
            H,
            0,
            out.stride(0),
            BH=c["BH"],
            num_warps=c["num_warps"],
        )
    return out


def _plane_reduce(leaf_reduce, ws, rank):
    """leaf_reduce in column-plane mode for [S, H] leaves: the planes go to leaf_reduce,
    this rank's plane comes back in [S, H] with +0 elsewhere."""

    def reduce(z):
        S, H = z.shape
        hp = H // ws
        red = leaf_reduce(z.view(S, ws, hp).transpose(0, 1).contiguous())
        full = torch.zeros_like(z)
        full[:, rank * hp : (rank + 1) * hp] = red
        return full

    return reduce


def _w2(
    act,
    w2,
    w2_scale,
    route,
    n_out,
    n_global,
    chunk_gather,
    block_m,
    ep=None,
):
    """bf16 [R, n_out]: the W2 product over the 8 fixed chunks of N with the pair tree
    (rounded once; with tensor parallelism after the plane gather)."""
    R, N_local = act.shape
    chunk = n_global // N_CHUNKS
    if chunk_gather is not None:
        ar = torch.arange(R, dtype=torch.int32, device=act.device)
        P = chunk_gather(
            _grouped_gemm(
                act, ar, w2, w2_scale, route, n_out, N_local // chunk, block_m
            )
        )
        assert P.shape[0] == N_CHUNKS, f"{P.shape[0]} chunk planes, expected {N_CHUNKS}"
        return _tree8_rows(
            P,
            torch.full((1,), R, dtype=torch.int64, device=act.device),
            torch.bfloat16,
        )
    assert N_local == n_global
    _, _, _, expt_offs, tile_offs, max_tiles = route
    # y is bf16: the combine reads fp32(bf16(y)) only. With ep, the leaves go to slots.
    out = torch.empty((R, n_out), dtype=torch.bfloat16, device=act.device)
    if R == 0:
        return out
    c, bk, is_async = _w2_plan(n_global, block_m, w2, w2_scale)
    nb = c["NUM_BUFFERS"] if is_async else 1
    pipe = {"NUM_BUFFERS": nb} if nb > 1 else {}
    if nb > 1:
        pipe.update(
            {
                k: c[k]
                for k in (
                    "N_FIRST",
                    "MMA_K",
                    "W_SWZ",
                )
                if k in c
            }
        )
    kernel = (
        _moe_mxfp4_bi_w2_tree_async_gluon_kernel
        if nb > 1
        else _moe_mxfp4_bi_w2_tree_gluon_kernel
    )
    zb = c["ZERO_ROWS"]
    if ep is not None:
        assert nb > 1
        slots, rows_tok, ord_, w_tok = ep
        planes = slots.dim() == 3  # [ws, slots, H / ws]
        ep_args = (
            slots,
            rows_tok,
            ord_,
            w_tok,
            rows_tok.numel(),
            max_tiles,
            slots.shape[2] if planes else n_out,
            slots.stride(0) if planes else 0,
        )
        pipe.update(
            EP_LEAF=True,
            ZB=zb,
            PLANES=planes,
        )
        n_tiles = max_tiles + triton.cdiv(rows_tok.numel(), zb)
    else:
        ep_args = (out,) * 4 + (0, max_tiles, n_out, 0) if nb > 1 else ()
        n_tiles = max_tiles
    grid = (n_tiles, triton.cdiv(n_out, c["BLOCK_N"]))
    kernel[grid[::-1] if pipe.get("N_FIRST") else grid](
        act,
        w2.view(torch.uint8),
        w2_scale.view(torch.uint8),
        out,
        expt_offs,
        tile_offs,
        w2.shape[0],
        n_out,
        chunk,
        act.stride(0),
        w2.stride(0),
        w2.stride(1),
        w2_scale.stride(0),
        w2_scale.stride(1),
        out.stride(0),
        *ep_args,
        BLOCK_M=block_m,
        BLOCK_N=c["BLOCK_N"],
        BLOCK_K=bk,
        num_warps=c["num_warps"],
        enable_fp_fusion=False,
        **pipe,
    )
    return out


def _combine_gluon(y, topk_weights, topk_ids, rows_tok, leaves=False):
    """bf16 [T, H]: top-16 combine; leaves=True: FP32 [T * 16, H] leaves by slot."""
    T = topk_ids.shape[0]
    H = y.shape[1]
    out = torch.empty(
        (T * TOP_K, H) if leaves else (T, H),
        dtype=torch.float32 if leaves else torch.bfloat16,
        device=y.device,
    )
    if T == 0:
        return out
    c = _config("combine")
    w_tok = topk_weights.to(torch.float32).reshape(T, TOP_K).contiguous()
    _moe_mxfp4_bi_combine_gluon_kernel[(T, triton.cdiv(H, c["BH"]))](
        y,
        w_tok,
        rows_tok,
        topk_ids.contiguous(),
        out,
        H,
        y.stride(0),
        out.stride(0),
        BH=c["BH"],
        num_warps=c["num_warps"],
        enable_fp_fusion=False,
        SCATTER=leaves,
    )
    return out


def _check_weights(x, w13, w13_scale, w2, w2_scale):
    E, N2 = w13.shape[0], w13.shape[1]
    H, N = x.shape[1], N2 // 2
    want = (
        (w13, (E, N2, H // 2)),
        (w13_scale, (E, N2, H // 32)),
        (w2, (E, H, N // 2)),
        (w2_scale, (E, H, N // 32)),
    )
    for t, shape in want:
        if tuple(t.shape) != shape or t.element_size() != 1 or t.stride(-1) != 1:
            raise ValueError(
                "moe_mxfp4_bi: weights must be 1-byte [E, 2N, H/2], [E, 2N, H/32], "
                "[E, H, N/2], [E, H, N/32] with unit last stride; got "
                f"{tuple(t.shape)} {t.dtype} stride {t.stride()}, want {shape}"
            )


def _block_tokens_max(x, w13, chunk_gather):
    """Most tokens per GEMM block that keep each buffer-op operand below 2**31 bytes."""
    H, N = x.shape[1], w13.shape[1] // 2
    row = max(2 * H, 2 * N, 4 * H if chunk_gather is not None else 0)
    return max(1, (2**31 - 1) // (TOP_K * row))


def _forward_block(
    x,
    w13,
    w13_scale,
    w2,
    w2_scale,
    topk_weights,
    topk_ids,
    situ_beta,
    situ_linear_beta,
    n_global,
    chunk_gather,
    expert_map,
    leaf_reduce,
    leaf_block,
    leaf_planes=None,
):
    T, H = x.shape
    E = w13.shape[0]
    bm = _block_m(T, E if expert_map is None else expert_map.numel())
    tok, rows_tok, expt_offs, tile_offs, max_tiles, ord_ = _route_fused(
        topk_ids, expert_map, E, bm
    )
    route = (None, None, tok, expt_offs, tile_offs, max_tiles)
    act = _gu_situ(x, tok, w13, w13_scale, route, situ_beta, situ_linear_beta, bm)
    ep_leaf = (
        leaf_reduce is not None
        and chunk_gather is None
        and T <= TOKEN_BLOCK_COLLECTIVE
        and _w2_async(n_global, bm, w2, w2_scale)
    )
    if ep_leaf:
        # EP up to 1024 tokens: W2 writes the leaves (+0 for other ranks) to slots.
        w_tok = topk_weights.to(torch.float32).reshape(T, TOP_K).contiguous()
        shape = (T * TOP_K, H)
        if leaf_planes is not None:
            shape = (leaf_planes[0], T * TOP_K, H // leaf_planes[0])
        slots = torch.empty(shape, dtype=torch.float32, device=x.device)
        _w2(
            act,
            w2,
            w2_scale,
            route,
            H,
            n_global,
            None,
            bm,
            ep=(slots, rows_tok, ord_, w_tok),
        )
        return _tree16_blocks(slots, topk_ids, leaf_reduce, leaf_block, leaf_planes)
    y = _w2(act, w2, w2_scale, route, H, n_global, chunk_gather, bm)
    if leaf_reduce is None:
        return _combine_gluon(y, topk_weights, topk_ids, rows_tok)
    # Expert parallel: leaves by slot, then leaf_reduce and tree per leaf_block.
    if leaf_planes is not None:
        leaf_reduce = _plane_reduce(leaf_reduce, *leaf_planes)
    slots = _combine_gluon(y, topk_weights, topk_ids, rows_tok, leaves=True)
    return _tree16_blocks(slots, topk_ids, leaf_reduce, leaf_block)


@torch.no_grad()
def moe_mxfp4_bi_forward(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    situ_beta: float,
    situ_linear_beta: float | None = None,
    n_global: int | None = None,
    chunk_gather: Callable[[torch.Tensor], torch.Tensor] | None = None,
    expert_map: torch.Tensor | None = None,
    leaf_reduce: Callable[[torch.Tensor], torch.Tensor] | None = None,
    token_block: int | None = None,
    leaf_planes: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Batch-invariant MXFP4 routed-expert MoE with the SiTU activation and top-16
    routing.

    For each slot (token t, expert e): gu = x_t @ deq(W13_e)^T (FP32, one bf16
    rounding); act = bf16(SiTU(gate, up)); y = act @ deq(W2_e)^T, summed over 8 fixed
    intermediate chunks with a fixed FP32 pair tree; leaf = w * fp32(bf16 y). The output
    is a fixed FP32 pair tree over the 16 leaves in expert-id order, rounded once to
    bf16 (-0 becomes +0). The bits of a token's output do not depend on the other
    tokens, on the batch size, on the order of the tokens, or on the tensor-parallel or
    expert-parallel layout. The call does not synchronize with the host, so it can be
    captured in a CUDA graph.

    Args:
        x: bf16 [T, H] token activations.
        w13: uint8 [E, 2N, H/2] MXFP4 gate and up weights (two E2M1 values per byte, low
            nibble first), rows [gate; up].
        w13_scale: uint8 [E, 2N, H/32] E8M0 scales of w13.
        w2: uint8 [E, H, N/2] MXFP4 down weights. w2_scale: uint8 [E, H, N/32].
        topk_weights: float [T, 16] routing weights. topk_ids: int [T, 16] expert ids;
            an id outside [0, E_global) and a slot whose expert_map value is not a local
            expert count as on another rank.
        situ_beta, situ_linear_beta: SiTU parameters (no linear part when
            situ_linear_beta is None).
        n_global: full intermediate size when this rank holds a tensor-parallel slice of
            N (default N).
        chunk_gather: tensor parallel only. Gathers FP32 [C_local, R, H] chunk planes of
            all ranks, in rank order.
        expert_map: expert parallel only. int [E_global]: local expert id, or -1 when
            the expert is on another rank.
        leaf_reduce: expert parallel only. FP32 sum of [T * 16, H] over the
            expert-parallel group.
        token_block: tokens per block of collective calls (default 1024): leaf_reduce
            and chunk_gather are called once per block.
            The result does not depend on this value. All ranks of a group must pass the
            same T, or their calls do not match.
        leaf_planes: (ws, rank), expert parallel only (opt-in). leaf_reduce gets the
            leaves as ws column planes [ws, T * 16, H / ws] and returns this rank's
            summed plane (a reduce-scatter). The result then holds this rank's H / ws
            columns and +0 in the others, with the same bits per element; the sum over
            the group is the full output.

    Returns:
        bf16 [T, H]. With expert or tensor parallelism, every rank of the group gets the
        same result (with leaf_planes, its own columns).

    Limits: top-k must be 16. N must be a multiple of 8 * 64 and H a multiple of 64.
    Weights are 1-byte tensors with a unit last stride. gfx950 only (pinned configs).
    """
    T = x.shape[0]
    if topk_ids.shape != (T, TOP_K):
        raise ValueError(
            f"moe_mxfp4_bi: topk_ids must be [{T}, {TOP_K}], "
            f"got {tuple(topk_ids.shape)}"
        )
    if x.dtype != torch.bfloat16:
        raise TypeError("moe_mxfp4_bi: x must be bf16")
    if topk_weights.shape != (T, TOP_K) or not topk_weights.is_floating_point():
        raise ValueError(f"moe_mxfp4_bi: topk_weights must be float [{T}, {TOP_K}]")
    ints = (torch.int32, torch.int64)
    if topk_ids.dtype not in ints or (
        expert_map is not None
        and (expert_map.dtype not in ints or expert_map.dim() != 1)
    ):
        raise TypeError("moe_mxfp4_bi: topk_ids and expert_map must be int32 or int64")
    n_global = n_global or w13.shape[1] // 2
    if (
        n_global % N_CHUNKS
        or (n_global // N_CHUNKS) % 64
        or (w13.shape[1] // 2) % (n_global // N_CHUNKS)
    ):
        raise ValueError(
            f"moe_mxfp4_bi: N={n_global} must split into {N_CHUNKS} chunks "
            "of a multiple of 64"
        )
    if x.shape[1] % 64:
        raise ValueError(f"moe_mxfp4_bi: H={x.shape[1]} must be a multiple of 64")
    _check_weights(x, w13, w13_scale, w2, w2_scale)
    if chunk_gather is not None and leaf_reduce is not None:
        raise ValueError("moe_mxfp4_bi: use tensor or expert parallelism, not both")
    if (expert_map is None) != (leaf_reduce is None):
        raise ValueError(
            "moe_mxfp4_bi: expert_map and leaf_reduce go together (expert parallel)"
        )
    if leaf_planes is not None:
        ws, rank = leaf_planes
        if (
            leaf_reduce is None
            or chunk_gather is not None
            or x.shape[1] % ws
            or not 0 <= rank < ws
        ):
            raise ValueError(
                f"moe_mxfp4_bi: leaf_planes={leaf_planes} needs expert "
                "parallelism and H % ws == 0"
            )
    x = x.contiguous()
    # Collectives run per collective block; without them the GEMM block is larger.
    c = _config("block_m")
    leaf_block = token_block or TOKEN_BLOCK_COLLECTIVE
    blk = token_block or (
        TOKEN_BLOCK_COLLECTIVE if chunk_gather is not None else c["TOKEN_BLOCK"]
    )
    cap = _block_tokens_max(x, w13, chunk_gather)
    if leaf_reduce is None and chunk_gather is None:
        # No collective calls: the block size does not change the bits.
        blk = min(blk, cap)
    else:
        if chunk_gather is None:
            blk = min(blk, cap - cap % leaf_block)
        if not 0 < blk <= cap:
            raise ValueError(
                f"moe_mxfp4_bi: token_block={leaf_block} is over {cap}, "
                "the op's limit"
            )
    assert blk % leaf_block == 0 or chunk_gather is None and leaf_reduce is None
    args = (w13, w13_scale, w2, w2_scale)
    rest = (
        situ_beta,
        situ_linear_beta,
        n_global,
        chunk_gather,
        expert_map,
        leaf_reduce,
    )
    if T <= blk:
        return _forward_block(
            x, *args, topk_weights, topk_ids, *rest, leaf_block, leaf_planes
        )

    return torch.cat(
        [
            _forward_block(
                x[i : i + blk],
                *args,
                topk_weights[i : i + blk],
                topk_ids[i : i + blk],
                *rest,
                leaf_block,
                leaf_planes,
            )
            for i in range(0, T, blk)
        ]
    )
