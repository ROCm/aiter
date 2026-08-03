"""Shared utilities for gfx1250 GEMM kernels (fp16 / mxfp4 / mxfp8)."""

import flydsl.expr as fx
from flydsl.expr import arith, gpu, rocdl, tdm_ops
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import T


def make_lds_copy_ops(bits):
    """Create one reusable layout/copy atom and return its load/store callables."""
    assert bits in (32, 64, 128), f"bits must be 32/64/128, got {bits}"
    elem_count = bits // fx.Int32.width
    layout = fx.make_layout(elem_count, 1)
    atom = fx.make_copy_atom(fx.UniversalCopy(bits), fx.Int32)
    ptr_ty = fx.PointerType.get(
        elem_ty=fx.Int32.ir_type,
        address_space=fx.AddressSpace.Shared,
        alignment=bits // 8,
    )

    def _view(lds_base_idx, byte_offset):
        byte_offset = fx.index_cast(T.index, byte_offset)
        addr_i32 = fx.index_cast(T.i32, lds_base_idx + byte_offset)
        ptr = fx.inttoptr(ptr_ty, addr_i32)
        return fx.Tensor(fx.make_view(ptr, layout))

    def load(lds_base_idx, byte_offset):
        rmem = fx.make_rmem_tensor(layout, fx.Int32)
        fx.copy_atom_call(atom, _view(lds_base_idx, byte_offset), rmem)
        return rmem.load()

    def store(lds_base_idx, byte_offset, data):
        rmem = fx.make_rmem_tensor(layout, fx.Int32)
        rmem.store(data)
        fx.copy_atom_call(atom, rmem, _view(lds_base_idx, byte_offset))

    return load, store


def workgroup_barrier(use_cluster=False):
    """Issue the appropriate barrier for LDS visibility.

    Cluster mode layers an inter-workgroup barrier on top of the regular
    workgroup barrier protocol, so call sites can treat it as a single
    "LDS is now readable" fence.
    """
    if use_cluster:
        cluster.cluster_barrier()
    else:
        gpu.barrier()


def pipeline_fence(outstanding=0, use_cluster=False):
    """Fused READY+REUSE fence for gfx1250 multi-buffer pipeline.

    Issues ``s_wait_tensorcnt`` followed by the appropriate barrier.
    """
    tdm_ops.tensor_wait(outstanding)
    workgroup_barrier(use_cluster=use_cluster)


import math as _math

LOG2E = _math.log2(_math.e)


def fmin_f32(a, b):
    """Scalar f32 min (maps to v_min_num_f32)."""
    import flydsl.expr as _fx

    return _fx.Float32(arith.minnumf(_raw(a), _raw(b)))


def fclamp_f32(x, lo, hi):
    """Scalar f32 clamp via v_med3_num_f32."""
    import flydsl.expr as _fx

    return _fx.Float32(rocdl.fmed3(T.f32, _raw(x), _raw(lo), _raw(hi)))


def fused_silu_swiglu_elem(g, u, *, swiglu, limit_f32, neg_limit_f32):
    """One (gate, up) pair -> fused silu or swiglu scalar (gpt-oss clamp)."""
    import flydsl.expr as _fx

    _one = _fx.Float32(1.0)
    g = fmin_f32(g, limit_f32)
    u = fclamp_f32(u, neg_limit_f32, limit_f32)
    if swiglu:
        nlog2e = _fx.Float32(-1.702 * LOG2E)
        exp_val = _fx.Float32(rocdl.exp2(T.f32, _raw(g * nlog2e)))
        sig = _fx.Float32(rocdl.rcp(T.f32, _one + exp_val))
        return g * sig * (u + _one)
    nlog2e = _fx.Float32(-LOG2E)
    exp_val = _fx.Float32(rocdl.exp2(T.f32, _raw(g * nlog2e)))
    sig = _fx.Float32(rocdl.rcp(T.f32, _one + exp_val))
    return g * sig * u


def batched_silu_swiglu(pairs, *, swiglu, limit_f32, neg_limit_f32, range_constexpr):
    """Batched silu/swiglu with pipelined exp2/rcp for better TRANS utilisation.

    Args:
        pairs: list of (gate, up) f32 value pairs.
        swiglu: True for swiglu, False for silu.
        limit_f32, neg_limit_f32: clamp bounds.
        range_constexpr: the FlyDSL ``range_constexpr`` helper.

    Returns:
        list of activated f32 values, same length as *pairs*.
    """
    import flydsl.expr as _fx

    _one = _fx.Float32(1.0)
    nlog2e = _fx.Float32((-1.702 * LOG2E) if swiglu else (-LOG2E))
    N = len(pairs)
    # Stage 1: clamp + exp2
    gs, us, exp_vals = [], [], []
    for i in range_constexpr(N):
        g = fmin_f32(pairs[i][0], limit_f32)
        u = fclamp_f32(pairs[i][1], neg_limit_f32, limit_f32)
        gs.append(g)
        us.append(u)
    rocdl.sched_barrier(0)
    for i in range_constexpr(N):
        exp_val = _fx.Float32(rocdl.exp2(T.f32, _raw(gs[i] * nlog2e)))
        exp_vals.append(exp_val)
    # Stage 2a: add 1+exp
    rocdl.sched_barrier(0)
    sum_vals = []
    for i in range_constexpr(N):
        sum_vals.append(_one + exp_vals[i])
    # Stage 2b: rcp
    rocdl.sched_barrier(0)
    rcp_vals = []
    for i in range_constexpr(N):
        rcp_vals.append(_fx.Float32(rocdl.rcp(T.f32, sum_vals[i])))
    # Stage 3: final mul
    rocdl.sched_barrier(0)
    results = []
    for i in range_constexpr(N):
        if swiglu:
            results.append(gs[i] * rcp_vals[i] * (us[i] + _one))
        else:
            results.append(gs[i] * rcp_vals[i] * us[i])
    return results


__all__ = [
    "LOG2E",
    "batched_silu_swiglu",
    "fclamp_f32",
    "fmin_f32",
    "fused_silu_swiglu_elem",
    "make_lds_copy_ops",
    "pipeline_fence",
    "workgroup_barrier",
]
