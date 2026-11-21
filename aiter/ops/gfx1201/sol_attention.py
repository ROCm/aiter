# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1201 Sol sparse attention for MiniMax-H3 (lossy, opt-in).

Block selection follows NVlabs Sol-Attn (``thresh_type="diag"``) with the Sol-H3 policy: per 64-row query block,
key block ``j`` is exact if ``qc . kc_j > qc . mu + tau * sqrt(sum qc^2 var + 1e-6)`` (``qc`` block-mean query,
``kc`` block-mean keys, ``mu``/``var`` their per-dimension mean/variance, scores in log2 units), if ``|qb - j| <= 1``,
or if the key or query block overlaps the first ``prefix`` tokens; larger ``tau`` is sparser. Each 512-row workgroup
computes the union of its eight query blocks exactly (a superset of the Sol-Attn selection) in the
list-driven ASM core (``hsa/gfx1201/sage_attention/attention_sol.hsaco``: the ``attention_pv_early`` core with a
per-workgroup KV tile list and an initial softmax state); all other blocks contribute through the Sol-Attn
block-summary correction (per row ``2^(q . kc_j)``, weighted by the block length, times the block's value sum; pooled
INT8 keys / FP8 value sums) computed by the HIP Phase-A kernel into that state.

Inputs are the ``prepare_sage`` / ``norm_rope_prepare_sage`` tuple ``(q_int8 [1,Sp,H,128], q_scale, k_int8,
k_scale, v_fp8 [1,H,128,Sp], v_scale)``. With every block selected the result is bitwise equal to
``launch_hip_sage_core``.
"""

import ctypes
import os

import torch
from torch import Tensor

from aiter.jit.core import AITER_ASM_DIR, compile_ops

KEY_BLOCK, WORKGROUP_ROWS, MAX_KEY_BLOCKS = 64, 512, 2048
STATE_SHAPE = (16, 33, 32, 4)  # per workgroup: waves x (32 o chunks + l/m chunk) x lanes x 4 floats


@compile_ops("module_gfx1201_sol_attention", fc_name="gfx1201_sol_route_hip", develop=True)
def gfx1201_sol_route_hip(
    q: Tensor,
    q_scale: Tensor,
    k: Tensor,
    k_scale: Tensor,
    v: Tensor,
    v_scale: Tensor,
    kbar: Tensor,
    vsum: Tensor,
    kb: Tensor,
    kb_scale: Tensor,
    vb: Tensor,
    vb_scale: Tensor,
    mu: Tensor,
    var: Tensor,
    tile_list: Tensor,
    tile_count: Tensor,
    mask: Tensor,
    valid: int,
    prefix: int,
    tau: float,
) -> None:
    """Pool K/V, quantize the pooled operands and write the per-workgroup tile list / Phase-A mask."""


@compile_ops("module_gfx1201_sol_attention", fc_name="gfx1201_sol_phase_a_hip", develop=True)
def gfx1201_sol_phase_a_hip(
    q: Tensor,
    q_scale: Tensor,
    v_scale: Tensor,
    kb: Tensor,
    vb: Tensor,
    kb_scale: Tensor,
    vb_scale: Tensor,
    cnt: Tensor,
    mask: Tensor,
    state: Tensor,
    valid: int,
    nkb: int,
    nkbp: int,
    n_wg: int,
) -> None:
    """Proxy attention over the unselected blocks -> initial (o, l, m) state of the ASM core."""


_runtime = None
_modules = {}
_plans = {}


def _check(status):
    if status:
        raise RuntimeError(f"HIP module API failed: {status}")


def _function():
    global _runtime
    if _runtime is None:
        _runtime = ctypes.CDLL("libamdhip64.so")
        _runtime.hipModuleLoad.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
        ]
        _runtime.hipModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        _runtime.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [
            ctypes.c_uint
        ] * 7 + [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
    code_object = os.path.join(
        AITER_ASM_DIR, "gfx1201", "sage_attention", "attention_sol.hsaco"
    )
    key = (torch.cuda.current_device(), code_object)
    if key not in _modules:
        module = ctypes.c_void_p()
        function = ctypes.c_void_p()
        _check(_runtime.hipModuleLoad(ctypes.byref(module), os.fsencode(key[1])))
        _check(
            _runtime.hipModuleGetFunction(
                ctypes.byref(function), module, b"sage_hip_attn_bm128_bn32"
            )
        )
        _modules[key] = (module, function)
    return _modules[key][1]


def _plan(device, padded, valid, heads):
    """Reusable device buffers of one (device, sequence, heads) shape."""
    key = (device, padded, valid, heads)
    if key not in _plans:
        nkb = (valid + KEY_BLOCK - 1) // KEY_BLOCK
        nkbp = (nkb + 31) // 32 * 32
        if nkbp > MAX_KEY_BLOCKS:
            raise ValueError(f"Sol attention supports up to {MAX_KEY_BLOCKS * KEY_BLOCK} tokens, got {valid}")
        n_wg = (valid + WORKGROUP_ROWS - 1) // WORKGROUP_ROWS * heads
        f32 = lambda *shape: torch.empty(*shape, device=device)
        cnt = torch.zeros(nkbp, device=device)
        cnt[:nkb] = KEY_BLOCK
        cnt[nkb - 1] = valid - (nkb - 1) * KEY_BLOCK
        stride = 2 * nkb + 2
        _plans[key] = {
            "nkb": nkb, "nkbp": nkbp, "n_wg": n_wg, "stride": stride, "cnt": cnt,
            "kbar": f32(heads, nkbp, 128),
            "vsum": f32(heads, 128, nkbp),
            "kb": torch.empty(1, nkbp, heads, 128, dtype=torch.int8, device=device),
            "kb_scale": f32(heads, nkbp // 32),
            "vb": torch.empty(heads, 128, nkbp, dtype=torch.uint8, device=device),
            "vb_scale": f32(heads, 128),
            "mu": f32(heads, 128),
            "var": f32(heads, 128),
            "tile_list": torch.empty(n_wg, stride, dtype=torch.int32, device=device),
            "tile_count": torch.empty(n_wg, dtype=torch.int32, device=device),
            "mask": torch.empty(n_wg, nkbp // 32, dtype=torch.int32, device=device),
            "state": torch.empty(n_wg, *STATE_SHAPE, device=device),
        }
    return _plans[key]


def sol_route(prepared, valid_seq_len, tau=1.0, prefix=0):
    """Run the router on the current stream; returns the plan (``tile_count`` holds the exact 32-key tiles of
    each workgroup, workgroup order ``head * ceil(valid / 512) + query_tile``)."""
    q, q_scale, k, k_scale, v, v_scale = prepared
    padded, heads = q.shape[1], q.shape[2]
    if padded != (valid_seq_len + 31) // 32 * 32 or q.dtype != torch.int8 or q.shape[0] != 1:
        raise ValueError("Expected the batch-1 prepare_sage tuple with the sequence padded to a multiple of 32")
    plan = _plan(q.device, padded, valid_seq_len, heads)
    gfx1201_sol_route_hip(
        q, q_scale, k, k_scale, v.view(torch.uint8), v_scale, plan["kbar"], plan["vsum"], plan["kb"],
        plan["kb_scale"], plan["vb"], plan["vb_scale"], plan["mu"], plan["var"], plan["tile_list"],
        plan["tile_count"], plan["mask"], valid_seq_len, prefix, float(tau),
    )
    return plan


def launch_hip_sol_core(prepared, out, valid_seq_len, plan):
    """List-driven ASM core: exact tiles of ``plan`` on top of the Phase-A state in ``plan["state"]``."""
    q, q_scale, k, k_scale, v, v_scale = prepared
    padded, heads = q.shape[1], q.shape[2]
    if out.dtype != torch.bfloat16 or out.device != q.device:
        raise ValueError("Expected BF16 output on the input device")
    with torch.cuda.device(q.device):
        function = _function()
        arguments = [
            ctypes.c_void_p(tensor.data_ptr())
            for tensor in (q, k, v, q_scale, k_scale, v_scale, out)
        ]
        arguments += [ctypes.c_int(value) for value in (1, padded, valid_seq_len, heads)]
        arguments += [
            ctypes.c_void_p(plan[name].data_ptr()) for name in ("tile_list", "tile_count", "state")
        ]
        arguments += [ctypes.c_int(plan["stride"]), ctypes.c_int(0)]
        parameters = (ctypes.c_void_p * len(arguments))(
            *[ctypes.addressof(argument) for argument in arguments]
        )
        _check(
            _runtime.hipModuleLaunchKernel(
                function,
                plan["n_wg"],
                1,
                1,
                512,
                1,
                1,
                0,
                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                parameters,
                None,
            )
        )


def sol_attention(prepared, out, valid_seq_len, tau=1.0, prefix=0):
    """Sol sparse attention into ``out`` (BF16 ``[1, Sp, H, 128]``, same layout as ``launch_hip_sage_core``).

    tau: Sol-Attn threshold in standard deviations (higher = sparser; Sol-H3 default 1.0); prefix: number of leading
    tokens computed exactly for every query (and queries inside it attend densely), i.e. the Sol-H3 sink.
    Returns the plan of this call.
    """
    q, q_scale = prepared[0], prepared[1]
    plan = sol_route(prepared, valid_seq_len, tau, prefix)
    gfx1201_sol_phase_a_hip(
        q, q_scale, prepared[5], plan["kb"], plan["vb"], plan["kb_scale"], plan["vb_scale"], plan["cnt"],
        plan["mask"], plan["state"], valid_seq_len, plan["nkb"], plan["nkbp"], plan["n_wg"],
    )
    launch_hip_sol_core(prepared, out, valid_seq_len, plan)
    return plan
