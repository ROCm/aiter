# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level warp-decode MoE MLP kernels for AMD GPU decode inference.

Warp-decode assigns one GPU wavefront (64 lanes) to compute one output scalar,
using ``v_dot2_f32_bf16`` instead of MFMA.  It is designed for small batch sizes
(B = 1–4 tokens), where a dense 16×16 MFMA tile is ≥75% empty.

Public API
----------
flydsl_wd_moe_gate_up(x, w_gate, w_up, router_ids, B, topk, inter, hidden, experts, ...)
    Stage 1: silu(x @ W_gate) * (x @ W_up) → inter_out [B*TOPK, INTER] bf16

flydsl_wd_moe_down_reduce(inter_out, w_down, router_ids, router_wts,
                          B, topk, inter, hidden, experts, ...)
    Stage 2: sum_k(router_wt_k * (inter_k @ W_down_k)) → y [B, HIDDEN] f32

Both functions auto-detect the GPU architecture and select the optimal
weight dtype (FP4 > FP8 > BF16) and compute path (dot2 vs scalar).

Tensor layouts
--------------
x          : [B, hidden]          bf16
w_gate     : [E*inter, hidden]    bf16 or uint8 (FP8 raw bytes) or uint8 (FP4)
w_up       : [E*inter, hidden]    same as w_gate
w_down     : [E*hidden, inter]    bf16 or uint8 (FP8 raw bytes)
router_ids : [B*topk]             int32
router_wts : [B*topk]             float32
inter_out  : [B*topk, inter]      bf16
y_out      : [B, hidden]          float32  (caller-zeroed before each call)
"""

import functools
from typing import Optional

import torch

from flydsl.runtime.device import get_rocm_arch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_WAVE_SIZE = 64


@functools.lru_cache(maxsize=1)
def _is_gfx950(arch: Optional[str] = None) -> bool:
    a = arch or get_rocm_arch()
    return a.startswith("gfx950") or a.startswith("gfx95")


def _best_n_waves(inter: int) -> int:
    """Return the best n_waves for cooperative LDS caching of inter_states.

    Rules derived from gfx942 and gfx950 benchmarks:
    - n_waves=4 gives 10-17 % speedup for inter >= 1536 at B >= 2.
    - n_waves=2 gives 5-10 % for 512 < inter < 1536.
    - n_waves=1 for inter <= 512 (grid already small; LDS overhead dominates).
    Constraint: inter must be divisible by n_waves * WAVE_SIZE * 2.
    """
    for nw in [4, 2, 1]:
        if inter % (nw * _WAVE_SIZE * 2) == 0:
            return nw
    return 1


def _ptr(t: torch.Tensor):
    """Convert a GPU tensor to a raw FlyDSL Uint8 pointer."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def _get_compile_fns():
    """Lazy-import the kernel builders.

    Tries the package-relative import first (normal installed path).
    Falls back to loading by file path when the module is loaded
    standalone via importlib (e.g. when aiter's C extension is stale).
    """
    try:
        from aiter.ops.flydsl.kernels.moe_warp_decode import (
            compile_wd_moe_gate_up,
            compile_wd_moe_down_reduce,
        )
        return compile_wd_moe_gate_up, compile_wd_moe_down_reduce
    except (ImportError, AttributeError):
        pass

    # Fallback: load directly from the source file next to this module.
    import importlib.util
    import pathlib

    _kernel_path = pathlib.Path(__file__).parent / "kernels" / "moe_warp_decode.py"
    _spec = importlib.util.spec_from_file_location("moe_warp_decode", _kernel_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod.compile_wd_moe_gate_up, _mod.compile_wd_moe_down_reduce


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def flydsl_wd_moe_gate_up(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    router_ids: torch.Tensor,
    B: int,
    topk: int,
    inter: int,
    hidden: int,
    experts: int,
    *,
    w_scale: float = 1.0,
    w_dtype: Optional[str] = None,
    stream: Optional[torch.cuda.Stream] = None,
    inter_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Warp-decode gate_up projection: silu(x @ W_gate) * (x @ W_up).

    Parameters
    ----------
    x          : [B, hidden] bf16 activations.
    w_gate     : [E*inter, hidden] weight matrix (bf16 or uint8 for fp8/fp4).
    w_up       : [E*inter, hidden] same dtype as w_gate.
    router_ids : [B*topk] int32, expert index for each (token, slot) pair.
    B, topk, inter, hidden, experts : shape scalars.
    w_scale    : per-tensor weight scale (only used for fp8/fp4 paths).
    w_dtype    : "bf16", "fp8", or "fp4"; None = auto-select by architecture.
    stream     : CUDA stream; None uses the current stream.
    inter_out  : optional pre-allocated [B*topk*inter] bf16 output buffer.

    Returns
    -------
    inter_out  : [B*topk*inter] bf16 (flat; reshape to [B*topk, inter]).
    """
    arch = get_rocm_arch()
    on_950 = _is_gfx950(arch)

    # Auto-select weight dtype based on architecture.
    if w_dtype is None:
        if on_950:
            w_dtype = "fp4" if w_gate.dtype == torch.uint8 else "fp8"
        else:
            w_dtype = "bf16"

    # FP4/FP8 require gfx950; silently fall back to BF16 on gfx942.
    if not on_950 and w_dtype in ("fp8", "fp4"):
        w_dtype = "bf16"

    use_dot2 = on_950  # v_dot2_f32_bf16 is gfx950-only

    compile_gate_up, _ = _get_compile_fns()
    exe = compile_gate_up(
        hidden=hidden, inter=inter, experts=experts, topk=topk,
        w_dtype=w_dtype, use_dot2=use_dot2,
    )

    if inter_out is None:
        inter_out = torch.zeros(B * topk * inter, dtype=torch.bfloat16, device=x.device)

    if stream is None:
        stream = torch.cuda.current_stream()

    exe(
        _ptr(inter_out), _ptr(x), _ptr(w_gate), _ptr(w_up),
        _ptr(router_ids), B, topk, inter, hidden, experts, w_scale, stream,
    )
    return inter_out


def flydsl_wd_moe_down_reduce(
    inter_out: torch.Tensor,
    w_down: torch.Tensor,
    router_ids: torch.Tensor,
    router_wts: torch.Tensor,
    B: int,
    topk: int,
    inter: int,
    hidden: int,
    experts: int,
    *,
    w_scale: float = 1.0,
    w_dtype: Optional[str] = None,
    h_per_warp: int = 2,
    n_waves: Optional[int] = None,
    stream: Optional[torch.cuda.Stream] = None,
    y_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Warp-decode down projection: sum_k(rw_k * (inter_k @ W_down_k)).

    Parameters
    ----------
    inter_out  : [B*topk, inter] bf16 activations (output of gate_up).
    w_down     : [E*hidden, inter] weight matrix (bf16 or uint8 for fp8).
    router_ids : [B*topk] int32, expert index for each (token, slot) pair.
    router_wts : [B*topk] float32, router weight per (token, slot) pair.
    B, topk, inter, hidden, experts : shape scalars.
    w_scale    : per-tensor weight scale for fp8 path.
    w_dtype    : "bf16" or "fp8"; None = auto-select by architecture.
    h_per_warp : 1 or 2 (H2 computes two adjacent output channels per wave;
                 2 is faster for B >= 2 and is the default).
    n_waves    : number of waves per block for LDS inter-state caching;
                 None = auto-select based on inter dimension.
    stream     : CUDA stream; None uses the current stream.
    y_out      : optional pre-allocated [B, hidden] float32 output buffer.
                 Must be zeroed before each call (accumulation via atomicAdd).

    Returns
    -------
    y_out : [B, hidden] float32.  Callers must zero this buffer before calling
            when reusing across different MoE layers.
    """
    arch = get_rocm_arch()
    on_950 = _is_gfx950(arch)

    if w_dtype is None:
        w_dtype = "fp8" if on_950 else "bf16"
    if not on_950 and w_dtype == "fp8":
        w_dtype = "bf16"

    use_dot2 = on_950

    if n_waves is None:
        n_waves = _best_n_waves(inter)

    _, compile_down = _get_compile_fns()
    exe = compile_down(
        hidden=hidden, inter=inter, experts=experts, topk=topk,
        use_dot2=use_dot2, h_per_warp=h_per_warp, w_dtype=w_dtype,
        n_waves=n_waves,
    )

    if y_out is None:
        y_out = torch.zeros(B, hidden, dtype=torch.float32, device=inter_out.device)

    if stream is None:
        stream = torch.cuda.current_stream()

    # inter_out may be [B*topk, inter] or [B*topk*inter] — flatten for _ptr.
    inter_flat = inter_out.reshape(-1)
    exe(
        _ptr(y_out), _ptr(inter_flat), _ptr(w_down),
        _ptr(router_ids), _ptr(router_wts),
        B, topk, inter, hidden, experts, w_scale, stream,
    )
    return y_out
