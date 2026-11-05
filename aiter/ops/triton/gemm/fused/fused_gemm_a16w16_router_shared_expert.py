# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import torch
import triton
import triton.experimental.gluon.language as gl

from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

_CONFIG_NAME = "FUSED-GEMM-A16W16-ROUTER-SHARED-EXPERT"
_TIMEOUT_NS = 20_000_000
_WS_MAX_MB = 1024
# Workspaces by device: ws[0] counts poll timeouts, then the M block counters
# (see the kernel). The kernel leaves the counters at zero.
_WORKSPACES: dict = {}
_PLANS: dict = {}
_NUM_CUS: dict = {}


def _get_config(M: int):
    config, _ = get_gemm_config(_CONFIG_NAME, M, backend="gluon")
    return config


def _wmma_layouts(warps_m: int, warps_n: int):
    warp_bases = [(0, 1 << i) for i in range(int(math.log2(warps_n)))]
    warp_bases += [(1 << i, 0) for i in range(int(math.log2(warps_m)))]
    wmma = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=tuple(warp_bases),
        instr_shape=[16, 16, 32],
    )
    op_a = gl.DotOperandLayout(operand_index=0, parent=wmma, k_width=8)
    op_b = gl.DotOperandLayout(operand_index=1, parent=wmma, k_width=8)
    return wmma, op_a, op_b


def _workspace(device: torch.device) -> torch.Tensor:
    ws = _WORKSPACES.get(device)
    if ws is None:
        ws = torch.zeros(2 + 2 * _WS_MAX_MB, dtype=torch.int32, device=device)
        _WORKSPACES[device] = ws
    return ws


def _num_cus(device: torch.device) -> int:
    n = _NUM_CUS.get(device)
    if n is None:
        n = torch.cuda.get_device_properties(device).multi_processor_count
        _NUM_CUS[device] = n
    return n


def _plan(M: int, nr: int, ngu: int, nd: int, config: dict, num_cus: int):
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (
        create_shared_layouts,
    )
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.fused.fused_gemm_a16w16_router_shared_expert import (
        _gemm_a16w16_router_shared_expert_kernel,
    )

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
    num_mb = triton.cdiv(M, bm)
    tiles = num_mb * (ngu // bn1 + triton.cdiv(nr, bn1) + nd // bn2)
    programs = min(tiles, num_cus * config["PROGRAMS_PER_CU"])
    meta = {
        "BLOCK_M": bm,
        "BLOCK_N1": bn1,
        "BLOCK_K1": bk1,
        "NUM_BUFFERS1": nb1,
        "BLOCK_N2": bn2,
        "BLOCK_K2": bk2,
        "NUM_PROGRAMS": programs,
        "NUM_WARPS": num_warps,
        "LAYOUT": "TN",
        "SHARED_A1": sa1,
        "SHARED_B1": sb1,
        "SHARED_A2": sa2,
        "SHARED_B2": sb2,
        "WMMA1": wmma1,
        "OP_A1": op_a1,
        "OP_B1": op_b1,
        "WMMA2": wmma2,
        "OP_A2": op_a2,
        "OP_B2": op_b2,
        "STORE1": store1,
        "TIMEOUT_NS": _TIMEOUT_NS,
        "num_warps": num_warps,
        "waves_per_eu": config.get("waves_per_eu", 0),
        "H_CACHE": config.get("H_CACHE", ".cg"),
    }
    return _gemm_a16w16_router_shared_expert_kernel, programs, num_mb, meta


def fused_gemm_a16w16_router_shared_expert_supported(
    M: int,
    x: torch.Tensor,
    w_router: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
) -> bool:
    """Whether fused_gemm_a16w16_router_shared_expert runs these operands."""
    return (
        get_arch() == "gfx1250"
        and x.dtype == torch.bfloat16
        and w_router.dtype == w_gate_up.dtype == w_down.dtype == torch.bfloat16
        and x.dim() == 2
        and x.stride(1) == 1
        and w_router.is_contiguous()
        and w_gate_up.is_contiguous()
        and w_down.is_contiguous()
        and tuple(w_router.shape[1:]) == (x.shape[1],)
        and tuple(w_gate_up.shape[1:]) == (x.shape[1],)
        and w_down.shape[1] * 2 == w_gate_up.shape[0]
        and _get_config(M) is not None
    )


def fused_gemm_a16w16_router_shared_expert(
    x: torch.Tensor,
    w_router: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    logits: torch.Tensor | None = None,
    shared: torch.Tensor | None = None,
    config: dict | None = None,
    n_router: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The MoE router GEMM and the bf16 shared expert in one gfx1250 kernel.

    Computes ``logits = x @ w_router.T`` and
    ``shared = (silu(x @ gate.T) * (x @ up.T)) @ w_down.T``. ``w_gate_up`` is
    ``gemm_a16w16_gated_interleave([gate; up])``. Each output is bitwise the
    output of its separate a16w16 GEMM (``gemm_a16w16_gated`` for gate / up).

    Args:
        x: [M, K] bf16, unit stride along K.
        w_router: [R, K] bf16 router weight, R >= n_router. Zero rows past
            n_router up to a multiple of 16 rows make the last router tile
            fast; a 257-row weight is correct but slower.
        w_gate_up: [2 * I, K] bf16, gate / up rows interleaved.
        w_down: [ND, I] bf16 down weight.
        logits: optional [M, n_router] bf16 output.
        shared: optional [M, ND] bf16 output.
        config: optional tile config; else the tuned config of M.
        n_router: the logits to compute (default R).

    Returns:
        (logits, shared).

    Limits: gfx1250; I a multiple of BLOCK_K2, 2 * I of BLOCK_N1, ND of
    BLOCK_N2; at most 1024 M blocks. The grid is at most one program per CU
    (PROGRAMS_PER_CU), so all programs are resident; a down tile waits for
    the gate / up tiles of its M block.
    """
    M, K = x.shape
    nr_rows, ngu, nd = w_router.shape[0], w_gate_up.shape[0], w_down.shape[0]
    nr = nr_rows if n_router is None else n_router
    if config is None:
        config = _get_config(M)
    key = (tuple(sorted(config.items())), M, nr, ngu, nd, x.device)
    plan = _PLANS.get(key)
    if plan is None:
        plan = _plan(M, nr, ngu, nd, config, _num_cus(x.device))
        _PLANS[key] = plan
    kernel, programs, num_mb, meta = plan
    if logits is None:
        logits = torch.empty((M, nr), dtype=x.dtype, device=x.device)
    if shared is None:
        shared = torch.empty((M, nd), dtype=x.dtype, device=x.device)
    h = torch.empty((M, ngu // 2), dtype=x.dtype, device=x.device)
    assert num_mb <= _WS_MAX_MB
    ws = _workspace(x.device)
    kernel[(programs, 1, 1)](
        x,
        w_router,
        w_gate_up,
        w_down,
        logits,
        h,
        shared,
        ws,
        M,
        x.stride(0),
        logits.stride(0),
        h.stride(0),
        shared.stride(0),
        K=K,
        NR=nr,
        NR_ROWS=nr_rows,
        NGU=ngu,
        ND=nd,
        **meta,
    )
    return logits, shared
