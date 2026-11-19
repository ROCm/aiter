# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Decode rewrite of ``fused_gemm_a16w16_router_shared_expert_topk`` (router GEMM + bf16 shared expert +
MoE top-k / routing / MXFP4 activation quant, <= 16 rows; 17 .. 48 rows with ATOM_FOLD_ROUTER_MAX_ROWS): same
signature, same outputs bit for bit, about 2 us less a launch on gfx1250 (3 .. 3.7 us less than the native pair at
20 .. 48 rows). See the kernel file for the schedule."""

import os

import torch
import triton.experimental.gluon.language as gl

from aiter.ops.triton.gemm.fused.fused_gemm_a16w16_router_shared_expert import (
    _TIMEOUT_NS,
    _num_cus,
    _wmma_layouts,
    _workspace,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

TOPK = 8
EXPERTS = 256
FEAT = 2048
MAX_ROWS = 16
# ATOM_FOLD_ROUTER_MAX_ROWS (default 16): rows up to which the kernel serves; 17 .. 48 take the wide variant
# (_gemm_a16w16_router_shared_expert_topk_v2w_kernel: M blocks of 16, multi-word token masks), e.g. 48 for the
# verify step of up to 12 requests x 4 tokens.
WIDE_MAX_ROWS = 48  # 4 M blocks (49 .. 64 rows) do not fit the LDS (349 KB > 320 KB)
LSC_ROWS = WIDE_MAX_ROWS


def _max_rows() -> int:
    raw = os.environ.get("ATOM_FOLD_ROUTER_MAX_ROWS", "")
    return max(1, min(WIDE_MAX_ROWS, int(raw))) if raw.strip() else MAX_ROWS


LSC_STRIDE = 272
# routes a route tile, by rows (measured: one tile a route at 4 rows, 4 routes a tile above)
_RPT = {4: 1}
_DEFAULT_RPT = 4
_RPT_WIDE = 4  # routes a route tile above 16 rows (measured: 2 and 8 slower at 20 .. 48 rows)
_FWS: dict = {}
_LSC: dict = {}
_PLANS: dict = {}


def _state(device):
    """This kernel's own route workspace (int32 [1024]; zero between launches) and logits flag scratch
    ([48, 272] bf16, all 0xFFFF between launches); each launch's last route tile restores both."""
    if device not in _FWS:
        _FWS[device] = torch.zeros(1024, dtype=torch.int32, device=device)
        _LSC[device] = torch.full((LSC_ROWS, LSC_STRIDE), -1, dtype=torch.int16, device=device).view(torch.bfloat16)
    return _FWS[device], _LSC[device]


def _plan(M, nr, ngu, nd, num_cus):
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import create_shared_layouts

    wide = M > MAX_ROWS
    mb = -(-M // 16)  # M blocks (wide variant)
    # wide: gate/up units of 16 weight rows, each on half of the M blocks
    bm, bn1, bk1, bkt1, rows_r, rows_g = 16, 16, 256, 512, 8, (16 if wide else 8)
    bn2, bk2, bkt2 = 16, 128, 512
    if ngu % (2 * rows_g) or (ngu // 2) % bkt2 or nd % bn2 or FEAT % bkt1:
        return None
    wmma1, op_a1, op_b1 = _wmma_layouts(1, 1)
    wmma2, op_a2, op_b2 = _wmma_layouts(1, 1)
    sa1, sb1 = create_shared_layouts(bm, bn1, bkt1, "TN")
    sa2, sb2 = create_shared_layouts(bm, bn2, bkt2, "TN")
    store1 = gl.BlockedLayout([1, 8], [32, 1], [1, 1], [1, 0])
    n_r = -(-nr // rows_r)
    n_ru = n_r * mb if wide else n_r  # router units (wide: one a router-row slice and M block)
    n_gu = ngu // rows_g * (2 if wide else 1)  # gate/up programs
    n_d = nd // bn2
    rpt0 = _RPT_WIDE if wide else _RPT.get(M, _DEFAULT_RPT)
    for rpt in (rpt0, 2, 4, 8):
        if rpt < rpt0:
            continue
        nq = M * (TOPK // rpt)
        pq0 = max(n_ru, nq)
        P = max(pq0 + n_gu, n_d + nq)
        if P <= num_cus:
            break
    else:
        return None
    meta = dict(
        BLOCK_M=bm, BLOCK_N1=bn1, BLOCK_K1=bk1, BKT1=bkt1, ROWS_R=rows_r, ROWS_G=rows_g, BLOCK_N2=bn2,
        BLOCK_K2=bk2, BKT2=bkt2, P=P, PQ0=pq0, NUM_WARPS=1, SHARED_A1=sa1, SHARED_B1=sb1, SHARED_A2=sa2,
        SHARED_B2=sb2, WMMA1=wmma1, OP_A1=op_a1, OP_B1=op_b1, WMMA2=wmma2, OP_A2=op_a2, OP_B2=op_b2,
        STORE1=store1, TIMEOUT_NS=_TIMEOUT_NS, H_CACHE=".cg", LSC_STRIDE=LSC_STRIDE, RPT=rpt, num_warps=1,
        waves_per_eu=0,
    )
    if wide:
        meta.update(MB=mb, WPE=1 if M <= 32 else 2)
    return P, meta


def _get_plan(M, nr, ngu, nd, device):
    key = (M, nr, ngu, nd, device)
    if key not in _PLANS:
        _PLANS[key] = _plan(M, nr, ngu, nd, _num_cus(device))
    return _PLANS[key]


def fused_gemm_a16w16_router_shared_expert_topk_v2_supported(
    x: torch.Tensor,
    w_router: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    n_router: int,
    align_m: int,
) -> bool:
    """Whether the v2 kernel serves this call: gfx1250, bf16, <= 16 rows (<= ATOM_FOLD_ROUTER_MAX_ROWS, at most 48,
    when set; align_m >= rows only up to 16), 2048 columns, 256 routed experts
    + the shared gate (n_router 257, router weight padded to <= 272 rows), every program resident."""
    M = x.shape[0]
    return (
        get_arch() == "gfx1250"
        and 0 < M <= _max_rows()
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.is_contiguous()
        and x.shape[1] == FEAT
        and n_router == EXPERTS + 1
        and n_router <= w_router.shape[0] <= LSC_STRIDE
        and w_router.dtype == w_gate_up.dtype == w_down.dtype == torch.bfloat16
        and w_router.is_contiguous()
        and w_gate_up.is_contiguous()
        and w_down.is_contiguous()
        and (align_m >= M or M > MAX_ROWS)
        and align_m > 0
        and (align_m & (align_m - 1)) == 0
        and _get_plan(M, n_router, w_gate_up.shape[0], w_down.shape[0], x.device) is not None
    )


def fused_gemm_a16w16_router_shared_expert_topk_v2(
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
    """As ``fused_gemm_a16w16_router_shared_expert_topk``: ``(logits, shared, routed)``, bitwise the same."""
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.fused.fused_gemm_a16w16_router_shared_expert_topk_v2 import (
        _gemm_a16w16_router_shared_expert_topk_v2_kernel,
        _gemm_a16w16_router_shared_expert_topk_v2w_kernel,
    )

    M, K = x.shape
    dev = x.device
    nr_rows, ngu, nd = w_router.shape[0], w_gate_up.shape[0], w_down.shape[0]
    P, meta = _get_plan(M, n_router, ngu, nd, dev)
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
    fws, lsc = _state(dev)
    has_nvr = num_valid_rows is not None
    nvr = num_valid_rows.reshape(-1)[:1].to(torch.int32) if has_nvr else fws
    kernel = (
        _gemm_a16w16_router_shared_expert_topk_v2_kernel if M <= MAX_ROWS
        else _gemm_a16w16_router_shared_expert_topk_v2w_kernel
    )
    kernel[(P, 1, 1)](
        x, w_router, w_gate_up, w_down, logits, h, shared, ws, fws, lsc, nvr, tw, tid, rows, mm, ps,
        payload.view(torch.int32), scale.view(torch.int32), M, x.stride(0), logits.stride(0), h.stride(0),
        shared.stride(0), K=K, NR=n_router, NR_ROWS=nr_rows, NGU=ngu, ND=nd, ALIGN_M=align_m,
        WMMA_REP=wmma_rep, HAS_NVR=has_nvr, **meta,
    )
    return logits, shared, (tw, tid, mm, rows, ps, payload, scale)
