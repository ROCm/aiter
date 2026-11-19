# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The router + shared expert kernel with the MoE top-k / routing / MXFP4 activation quant
folded in (decode, <= 16 rows): one launch for
``fused_gemm_a16w16_router_shared_expert`` + ``fused_topk_route_quant`` (the grouped MoE's
deferred decode path), every output bitwise the pair's. See the kernel file for the schedule.
"""

import torch
import triton
import triton.experimental.gluon.language as gl

from aiter.ops.triton.gemm.fused.fused_gemm_a16w16_router_shared_expert import (
    _TIMEOUT_NS,
    _get_config,
    _num_cus,
    _wmma_layouts,
    _workspace,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

TOPK = 8
EXPERTS = 256
FEAT = 2048
MAX_ROWS = 16
_FOLD_WS: dict = {}
_PLANS: dict = {}


def _fold_workspace(device) -> torch.Tensor:
    """[1] router-wave arrivals, [3] route tiles counted out, [256:512] per-expert token masks;
    zero between launches (the kernel's last route tile restores that)."""
    ws = _FOLD_WS.get(device)
    if ws is None:
        ws = torch.zeros(512, dtype=torch.int32, device=device)
        _FOLD_WS[device] = ws
    return ws


def _plan(M, nr, ngu, nd, config, num_cus):
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import create_shared_layouts

    bm = config["BLOCK_M"]
    bn1, bk1, nb1 = config["BLOCK_N1"], config["BLOCK_K1"], config["NUM_BUFFERS1"]
    bn2, bk2 = config["BLOCK_N2"], config["BLOCK_K2"]
    num_warps = config["num_warps"]
    warps_n1 = min(config["WARPS_N1"], num_warps)
    warps_n2 = min(config["WARPS_N2"], num_warps)
    wmma1, op_a1, op_b1 = _wmma_layouts(num_warps // warps_n1, warps_n1)
    wmma2, op_a2, op_b2 = _wmma_layouts(num_warps // warps_n2, warps_n2)
    sa1, sb1 = create_shared_layouts(bm, bn1, bk1, "TN")
    sa2, sb2 = create_shared_layouts(bm, bn2, bk2, "TN")
    half = bn1 // 2
    vec = min(8, half)
    tpw_n = min(32, half // vec)
    store1 = gl.BlockedLayout([1, vec], [32 // tpw_n, tpw_n], [num_warps, 1], [1, 0])
    base = ngu // bn1 + triton.cdiv(nr, bn1) + nd // bn2
    # routes a route tile: the fewest that keep every program resident
    rpt = next((r for r in (1, 2, 4, 8) if base + M * (TOPK // r) <= num_cus), None)
    if rpt is None:
        return None
    tiles = base + M * (TOPK // rpt)
    meta = dict(
        BLOCK_M=bm, BLOCK_N1=bn1, BLOCK_K1=bk1, NUM_BUFFERS1=nb1, BLOCK_N2=bn2, BLOCK_K2=bk2,
        NUM_PROGRAMS=tiles, NUM_WARPS=num_warps, LAYOUT="TN", SHARED_A1=sa1, SHARED_B1=sb1,
        SHARED_A2=sa2, SHARED_B2=sb2, WMMA1=wmma1, OP_A1=op_a1, OP_B1=op_b1, WMMA2=wmma2,
        OP_A2=op_a2, OP_B2=op_b2, STORE1=store1, TIMEOUT_NS=_TIMEOUT_NS, num_warps=num_warps,
        waves_per_eu=config.get("waves_per_eu", 0), H_CACHE=config.get("H_CACHE", ".cg"), RPT=rpt,
    )
    return tiles, meta


def _get_plan(M, nr, ngu, nd, device):
    key = (M, nr, ngu, nd, device)
    if key not in _PLANS:
        config = _get_config(M)
        ok = config is not None and config["BLOCK_M"] >= M
        _PLANS[key] = _plan(M, nr, ngu, nd, config, _num_cus(device)) if ok else None
    return _PLANS[key]


def fused_gemm_a16w16_router_shared_expert_topk_supported(
    x: torch.Tensor,
    w_router: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    n_router: int,
    align_m: int,
) -> bool:
    """Whether the folded kernel serves this call: gfx1250, bf16, <= 16 rows (one M block of
    the decode config), 2048 columns, 256 routed experts + the shared gate (n_router 257),
    every program resident, and align_m >= rows."""
    M = x.shape[0]
    return (
        get_arch() == "gfx1250"
        and 0 < M <= MAX_ROWS
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.is_contiguous()
        and x.shape[1] == FEAT
        and n_router == EXPERTS + 1
        and w_router.dtype == w_gate_up.dtype == w_down.dtype == torch.bfloat16
        and w_router.is_contiguous()
        and w_gate_up.is_contiguous()
        and w_down.is_contiguous()
        and align_m >= M
        and (align_m & (align_m - 1)) == 0
        and _get_plan(M, n_router, w_gate_up.shape[0], w_down.shape[0], x.device) is not None
    )


def fused_gemm_a16w16_router_shared_expert_topk(
    x: torch.Tensor,
    w_router: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    n_router: int,
    *,
    align_m: int,
    contiguous_m: int,
    wmma_rep: int,
    num_valid_rows: torch.Tensor | None = None,
    logits: torch.Tensor | None = None,
    shared: torch.Tensor | None = None,
):
    """``(logits, shared, routed)``: ``logits``/``shared`` as
    ``fused_gemm_a16w16_router_shared_expert(x, w_router, w_gate_up, w_down, n_router=n_router)``
    and ``routed`` = (topk_weights, topk_ids, masked_m, topids_to_rows, psum, payload, scale) as
    ``fused_topk_route_quant(logits[:, :256], x, 8, align_m=, contiguous_m=, wmma_rep=,
    quant_mode="fp4", num_valid_rows=)`` returns it."""
    M, K = x.shape
    dev = x.device
    nr_rows, ngu, nd = w_router.shape[0], w_gate_up.shape[0], w_down.shape[0]
    tiles, meta = _get_plan(M, n_router, ngu, nd, dev)
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.fused.fused_gemm_a16w16_router_shared_expert_topk import (
        _gemm_a16w16_router_shared_expert_topk_kernel,
    )

    if logits is None:
        logits = torch.empty((M, n_router), dtype=x.dtype, device=dev)
    if shared is None:
        shared = torch.empty((M, nd), dtype=x.dtype, device=dev)
    h = torch.empty((M, ngu // 2), dtype=x.dtype, device=dev)
    tw = torch.empty((M, TOPK), dtype=torch.float32, device=dev)
    tid = torch.empty((M, TOPK), dtype=torch.int32, device=dev)
    rows = torch.empty((M, TOPK), dtype=torch.int32, device=dev)
    mm = torch.empty(EXPERTS, dtype=torch.int32, device=dev)
    ps = torch.empty(EXPERTS, dtype=torch.int32, device=dev)
    payload = torch.empty((1, contiguous_m, K // 2), dtype=torch.uint8, device=dev)
    scale = torch.empty((1, contiguous_m // wmma_rep, (K // 32) * wmma_rep), dtype=torch.uint8, device=dev)
    ws = _workspace(dev)
    fws = _fold_workspace(dev)
    has_nvr = num_valid_rows is not None
    nvr = num_valid_rows.reshape(-1)[:1].to(torch.int32) if has_nvr else fws
    _gemm_a16w16_router_shared_expert_topk_kernel[(tiles, 1, 1)](
        x, w_router, w_gate_up, w_down, logits, h, shared, ws, fws, nvr, tw, tid, rows, mm, ps,
        payload.view(torch.int32), scale.view(torch.int32), M, x.stride(0), logits.stride(0),
        h.stride(0), shared.stride(0),
        K=K, NR=n_router, NR_ROWS=nr_rows, NGU=ngu, ND=nd, ALIGN_M=align_m, WMMA_REP=wmma_rep,
        HAS_NVR=has_nvr, **meta,
    )
    return logits, shared, (tw, tid, mm, rows, ps, payload, scale)
