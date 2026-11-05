# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import torch
import triton
import triton.experimental.gluon.language as gl

from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

_CONFIG_NAME = "FUSED-GEMM-A16W16-ROUTER-SHARED-EXPERT"
# Above this many rows the persistent decode kernel is slower than separate
# GEMMs; those sizes take the prefill path (`fused_gemm_a16w16_router_gate_up`
# for the router and gate_up as one GEMM, then the down GEMM).
_DECODE_MAX_M = 512
# Prefill: rows of the router weight the router tiles cover; the rest (the
# shared-gate logit and the zero padding) go to the narrow tail tiles.
_NR_MAIN = 256
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
        and (
            _get_config(M) is not None
            if M <= _DECODE_MAX_M
            else _NR_MAIN < w_router.shape[0] <= _NR_MAIN + 16
            and w_gate_up.shape[0] % 256 == 0
        )
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

    Above 512 rows (prefill) the router and gate_up run as one GEMM
    (`fused_gemm_a16w16_router_gate_up`) and down as the a16w16 GEMM; the
    router weight then has 257 to 272 rows.

    Limits: gfx1250; I a multiple of BLOCK_K2, 2 * I of BLOCK_N1, ND of
    BLOCK_N2; at most 1024 M blocks. The grid is at most one program per CU
    (PROGRAMS_PER_CU), so all programs are resident; a down tile waits for
    the gate / up tiles of its M block.
    """
    M, K = x.shape
    if M > _DECODE_MAX_M and (
        config is None or "BLOCK_M1" in config or "MODE" in config
    ):
        return fused_gemm_a16w16_router_shared_expert_prefill(
            x, w_router, w_gate_up, w_down, logits, shared, config, n_router
        )
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


# ---------------------------------------------------------------------------
# Prefill: the router + gate_up GEMM (one persistent kernel), then down.
# ---------------------------------------------------------------------------

_PREFILL_PLANS: dict = {}
_PREFILL_COMPILED: dict = {}
# The fused kernel's config where the prefill table at M says another mode.
_FUSED_DEFAULT = {
    "BLOCK_M1": 256,
    "BLOCK_N1": 128,
    "BLOCK_K1": 128,
    "NUM_BUFFERS1": 3,
    "num_warps": 8,
    "waves_per_eu": 1,
}


def _get_prefill_config(M: int):
    config, _ = get_gemm_config(
        _CONFIG_NAME, M, specialized_filename="PREFILL", backend="gluon"
    )
    return config


def _prefill_programs(M: int, ngu: int, config: dict, num_cus: int):
    """The persistent grid: one program per tile up to PROGRAMS_PER_CU a CU."""
    bm1 = config["BLOCK_M1"]
    bm2, bn2 = config.get("BLOCK_M2", bm1), config.get("BLOCK_N2", 128)
    tiles = -(-M // bm1) * (ngu // config["BLOCK_N1"])
    tiles += -(-M // bm2) * (_NR_MAIN // bn2 + 1)
    return min(tiles, num_cus * config.get("PROGRAMS_PER_CU", 1))


def _prefill_plan(nr_rows: int, ngu: int, config: dict, programs: int):
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (
        create_shared_layouts,
    )
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.fused.fused_gemm_a16w16_router_shared_expert import (
        _gemm_a16w16_router_gate_up_kernel,
    )

    num_warps = config["num_warps"]
    bm1, bn1 = config["BLOCK_M1"], config["BLOCK_N1"]
    bk1, nb1 = config["BLOCK_K1"], config["NUM_BUFFERS1"]
    bm2, bn2 = config.get("BLOCK_M2", bm1), config.get("BLOCK_N2", 128)
    bk2, nb2 = config.get("BLOCK_K2", bk1), config.get("NUM_BUFFERS2", nb1)
    bnt = config.get("BLOCK_NT", 16)
    warps_n1 = min(config.get("WARPS_N1", 2), num_warps)
    warps_n2 = min(config.get("WARPS_N2", 2), num_warps)
    wmma1, op_a1, op_b1 = _wmma_layouts(num_warps // warps_n1, warps_n1)
    wmma2, op_a2, op_b2 = _wmma_layouts(num_warps // warps_n2, warps_n2)
    wmmat, op_at, op_bt = _wmma_layouts(num_warps, 1)
    sa1, sb1 = create_shared_layouts(bm1, bn1, bk1, "TN")
    sa2, sb2 = create_shared_layouts(bm2, bn2, bk2, "TN")
    _, sbt = create_shared_layouts(bm2, bnt, bk2, "TN")
    meta = {
        "BLOCK_M1": bm1,
        "BLOCK_N1": bn1,
        "BLOCK_K1": bk1,
        "NUM_BUFFERS1": nb1,
        "BLOCK_M2": bm2,
        "BLOCK_N2": bn2,
        "BLOCK_K2": bk2,
        "NUM_BUFFERS2": nb2,
        "BLOCK_NT": bnt,
        "NUM_PROGRAMS": programs,
        "H_ROWS": config.get("H_ROWS", 32),
        "NUM_WARPS": num_warps,
        "SHARED_A1": sa1,
        "SHARED_B1": sb1,
        "WMMA1": wmma1,
        "OP_A1": op_a1,
        "OP_B1": op_b1,
        "SHARED_A2": sa2,
        "SHARED_B2": sb2,
        "WMMA2": wmma2,
        "OP_A2": op_a2,
        "OP_B2": op_b2,
        "SHARED_BT": sbt,
        "WMMAT": wmmat,
        "OP_AT": op_at,
        "OP_BT": op_bt,
        "num_warps": num_warps,
        "waves_per_eu": config.get("waves_per_eu", 0),
    }
    return _gemm_a16w16_router_gate_up_kernel, meta


def _launch_router_gate_up(x, w_router, w_gate_up, logits, h, nr, config):
    M, K = x.shape
    nr_rows, ngu = w_router.shape[0], w_gate_up.shape[0]
    programs = _prefill_programs(M, ngu, config, _num_cus(x.device))
    # Not keyed by M: eager prefill steps come in every size.
    key = (tuple(sorted(config.items())), programs, nr, nr_rows, ngu)
    plan = _PREFILL_PLANS.get(key)
    if plan is None:
        assert ngu % config["BLOCK_N1"] == 0 and ngu % 2 == 0
        assert _NR_MAIN < nr <= nr_rows <= _NR_MAIN + config.get("BLOCK_NT", 16)
        plan = _prefill_plan(nr_rows, ngu, config, programs)
        _PREFILL_PLANS[key] = plan
    kernel, meta = plan
    args = (
        x,
        w_router,
        w_gate_up,
        logits,
        h,
        M,
        x.stride(0),
        logits.stride(0),
        h.stride(0),
    )
    # Eager prefill is host-bound: launch the compiled kernel directly under a
    # key at least as fine as Triton's specialization (int == 1 / % 16,
    # 16-byte aligned pointers); anything else, and a first launch, JITs.
    aligned = not any(t.data_ptr() % 16 for t in args[:5])
    ckey = (key, K, x.dtype, x.device) + tuple((v == 1, v % 16 == 0) for v in args[5:])
    compiled = _PREFILL_COMPILED.get(ckey) if aligned else None
    if compiled is not None:
        compiled[(programs, 1, 1)](*args)
        return
    compiled = kernel[(programs, 1, 1)](
        *args,
        K=K,
        NR=nr,
        NR_MAIN=_NR_MAIN,
        NR_ROWS=nr_rows,
        NGU=ngu,
        **meta,
    )
    if aligned and compiled is not None:
        _PREFILL_COMPILED[ckey] = compiled


def fused_gemm_a16w16_router_gate_up(
    x: torch.Tensor,
    w_router: torch.Tensor,
    w_gate_up: torch.Tensor,
    logits: torch.Tensor | None = None,
    h: torch.Tensor | None = None,
    config: dict | None = None,
    n_router: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The MoE router GEMM and the shared expert's gate_up GEMM + SiLU-and-mul
    as one gfx1250 GEMM over [gate_up; router] (prefill sizes).

    Returns ``(logits, h)``: ``logits = x @ w_router[:n_router].T`` [M,
    n_router] and ``h = silu(x @ gate.T) * (x @ up.T)`` [M, I], each bitwise
    the router GEMM's / ``gemm_a16w16_gated``'s. ``w_router`` [R, K] has R in
    (256, 256 + BLOCK_NT] rows (zero rows past n_router); ``w_gate_up`` is
    ``gemm_a16w16_gated_interleave([gate; up])``. ``config`` is a kernel tile
    config (BLOCK_M1 / BLOCK_N1 / ... ); default: the fused config of the
    prefill table at M, else of its "any" bucket.
    """
    M = x.shape[0]
    nr = w_router.shape[0] if n_router is None else n_router
    if config is None:
        config = _get_prefill_config(M)
        if config.get("MODE", "fused") != "fused":
            config = _FUSED_DEFAULT
    config = {k: v for k, v in config.items() if k not in ("MODE", "DOWN")}
    if logits is None:
        logits = torch.empty((M, nr), dtype=x.dtype, device=x.device)
    if h is None:
        h = torch.empty((M, w_gate_up.shape[0] // 2), dtype=x.dtype, device=x.device)
    _launch_router_gate_up(x, w_router, w_gate_up, logits, h, nr, config)
    return logits, h


def fused_gemm_a16w16_router_shared_expert_prefill(
    x, w_router, w_gate_up, w_down, logits=None, shared=None, config=None, n_router=None
):
    """`fused_gemm_a16w16_router_shared_expert` at prefill sizes, bitwise the
    router GEMM, gemm_a16w16_gated and the down GEMM. ``w_router`` has 257 to
    272 rows (zero rows past n_router).

    The prefill table picks per M (by graph-replay timings of the whole
    sequence): "fused", the router and gate_up as one GEMM
    (`fused_gemm_a16w16_router_gate_up`); "split", the router GEMM and
    gemm_a16w16_gated at their own tile configs (ROUTER / GATE_UP); or
    "separate", the GEMMs at their default configs. Then the down GEMM
    (`gemm_a16w16`, at DOWN if the table gives one)."""
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import (
        gemm_a16w16,
        gemm_a16w16_gated,
    )

    M = x.shape[0]
    nr = w_router.shape[0] if n_router is None else n_router
    if config is None:
        config = _get_prefill_config(M)
    if logits is None:
        logits = torch.empty((M, nr), dtype=x.dtype, device=x.device)
    if shared is None:
        shared = torch.empty((M, w_down.shape[0]), dtype=x.dtype, device=x.device)
    h = torch.empty((M, w_gate_up.shape[0] // 2), dtype=x.dtype, device=x.device)
    mode = config.get("MODE", "fused")
    if mode == "fused":
        kcfg = {k: v for k, v in config.items() if k not in ("MODE", "DOWN")}
        _launch_router_gate_up(x, w_router, w_gate_up, logits, h, nr, kcfg)
    elif mode == "split":
        gemm_a16w16(
            x,
            w_router[:nr],
            None,
            x.dtype,
            y=logits,
            config=dict(config["ROUTER"]),
            kernel_type="compute_bound",
        )
        gemm_a16w16_gated(
            x,
            w_gate_up,
            y=h,
            config=dict(config["GATE_UP"]),
            kernel_type="compute_bound",
        )
    elif mode == "separate":
        # Today's three GEMMs at their default configs.
        gemm_a16w16(x, w_router[:nr], None, x.dtype, y=logits)
        gemm_a16w16_gated(x, w_gate_up, y=h)
    else:
        raise ValueError(f"unknown prefill mode {mode!r}")
    down = config.get("DOWN")
    if down is None:
        gemm_a16w16(h, w_down, None, x.dtype, y=shared)
    else:
        gemm_a16w16(
            h,
            w_down,
            None,
            x.dtype,
            y=shared,
            config=dict(down),
            kernel_type="compute_bound",
        )
    return logits, shared
