"""Shared utilities for gfx1250 GEMM kernels (fp16 / mxfp4 / mxfp8)."""

from collections import namedtuple

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_dialect
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


def lds_addr_keepalive(*bases_idx):
    """Pin the given LDS base-address registers live to this program point.

    Under max VGPR pressure the allocator reuses a ds_load's base-address
    register as the *destination* of a later ds_load once that address is dead.
    When earlier loads off the base are still in flight, the overwrite is a WAR
    hazard and the backend gates it with ``s_wait_alu depctr_vm_vsrc(N)``.
    Emitting a side-effecting no-op that reads the bases extends their live
    ranges past those later loads, so the allocator must place the later
    destinations in fresh registers and the WAR wait disappears. Relies on
    there being VGPR headroom (occupancy already at the floor, no spills).

    Costs zero instructions: the asm body is empty, only the "v" constraints
    and the side-effect marker reach the allocator.
    """
    from flydsl.expr.arith import ArithValue as _AV

    ops = [_raw(arith.index_cast(T.i32, _AV(b))) for b in bases_idx]
    if not ops:
        return
    llvm_dialect.InlineAsmOp(
        res=None,
        operands_=ops,
        asm_string="; lds addr keepalive",
        constraints=",".join(["v"] * len(ops)),
        has_side_effects=True,
        is_align_stack=False,
    )


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


def _tanh_f32(x, tanh_mul):
    """tanh(x) via the sigmoid identity tanh(z) = 2*sigmoid(2z) - 1.

    ``tanh_mul`` is the caller-hoisted ``-2*log2(e)/beta`` multiplier, so this
    evaluates ``2*rcp(1 + exp2(x * tanh_mul)) - 1`` for ``tanh(x/beta)`` in one
    exp2 + one rcp. Saturating rather than branchy: a large positive argument
    drives exp2 to +inf and rcp(+inf) to 0 (-> -1), a large negative one drives
    exp2 to 0 (-> +1), so no |x| fixup or sign select is needed.
    """
    import flydsl.expr as _fx

    _one = _fx.Float32(1.0)
    _two = _fx.Float32(2.0)
    exp_val = _fx.Float32(rocdl.exp2(T.f32, _raw(x * tanh_mul)))
    rcp_val = _fx.Float32(rocdl.rcp(T.f32, _one + exp_val))
    return _two * rcp_val - _one


# Loop-invariant f32 multipliers for the SiTUv2 epilogue, hoisted out of the
# per-element math by situv2_consts().
SituV2Consts = namedtuple("SituV2Consts", "beta gate_tanh_mul linear_beta up_tanh_mul")


def situv2_consts(beta, linear_beta):
    """Fold the SiTUv2 betas into the per-element multipliers, once per kernel.

    The two reciprocals are taken here with v_rcp_f32 rather than passed in from
    the host: both are uniform across the tile, so this is two extra VALU ops per
    kernel, hoisted out of the inner loop, in exchange for two fewer kernel args
    and no way for a caller to hand in a beta and a reciprocal that disagree.
    v_rcp_f32's ~1 ulp sits far below the MXFP4 quantisation this feeds.

    Hoisting keeps the inner loop at 3 exp2 + 3 rcp per element.
    """
    import flydsl.expr as _fx

    neg_two_log2e = _fx.Float32(-2.0 * LOG2E)
    return SituV2Consts(
        beta=beta,
        gate_tanh_mul=neg_two_log2e * _fx.Float32(rocdl.rcp(T.f32, _raw(beta))),
        linear_beta=linear_beta,
        up_tanh_mul=neg_two_log2e * _fx.Float32(rocdl.rcp(T.f32, _raw(linear_beta))),
    )


def fused_situv2_elem(g, u, *, consts):
    """One (gate, up) pair -> SiTUv2 (Kimi-K3 hidden_act="situ").

        beta * tanh(g/beta) * sigmoid(g) * linear_beta * tanh(u/linear_beta)

    ``consts`` comes from situv2_consts(). No clamp: SiTUv2 is bounded by
    construction, so the swiglu limit does not apply.
    """
    import flydsl.expr as _fx

    _one = _fx.Float32(1.0)
    nlog2e = _fx.Float32(-LOG2E)
    exp_val = _fx.Float32(rocdl.exp2(T.f32, _raw(g * nlog2e)))
    sig = _fx.Float32(rocdl.rcp(T.f32, _one + exp_val))
    gate_act = consts.beta * _tanh_f32(g, consts.gate_tanh_mul) * sig
    up_act = consts.linear_beta * _tanh_f32(u, consts.up_tanh_mul)
    return gate_act * up_act


def batched_situv2(pairs, *, consts, range_constexpr):
    """Batched SiTUv2 with pipelined exp2/rcp for better TRANS utilisation.

    Same staging idea as batched_silu_swiglu, over the three transcendental
    pairs SiTUv2 needs per element: sigmoid(gate), tanh(gate/beta) and
    tanh(up/linear_beta). Grouping all exp2s, then all rcps, keeps the TRANS
    unit busy instead of stalling on each dependent pair in turn.

    Args:
        pairs: list of (gate, up) f32 value pairs.
        consts: SituV2Consts from situv2_consts().
        range_constexpr: the FlyDSL ``range_constexpr`` helper.

    Returns:
        list of activated f32 values, same length as *pairs*.
    """
    import flydsl.expr as _fx

    _one = _fx.Float32(1.0)
    _two = _fx.Float32(2.0)
    nlog2e = _fx.Float32(-LOG2E)
    N = len(pairs)
    # Stage 1: all exp2 arguments, then all exp2.
    args = []
    for i in range_constexpr(N):
        g, u = pairs[i]
        args.append(g * nlog2e)  # sigmoid(gate)
        args.append(g * consts.gate_tanh_mul)  # tanh(gate/beta)
        args.append(u * consts.up_tanh_mul)  # tanh(up/linear_beta)
    rocdl.sched_barrier(0)
    exp_vals = []
    for i in range_constexpr(3 * N):
        exp_vals.append(_fx.Float32(rocdl.exp2(T.f32, _raw(args[i]))))
    # Stage 2a: 1 + exp
    rocdl.sched_barrier(0)
    sum_vals = []
    for i in range_constexpr(3 * N):
        sum_vals.append(_one + exp_vals[i])
    # Stage 2b: rcp
    rocdl.sched_barrier(0)
    rcp_vals = []
    for i in range_constexpr(3 * N):
        rcp_vals.append(_fx.Float32(rocdl.rcp(T.f32, sum_vals[i])))
    # Stage 3: sigmoid / tanh assembly and the final product.
    rocdl.sched_barrier(0)
    results = []
    for i in range_constexpr(N):
        sig = rcp_vals[3 * i]
        gate_tanh = _two * rcp_vals[3 * i + 1] - _one
        up_tanh = _two * rcp_vals[3 * i + 2] - _one
        gate_act = consts.beta * gate_tanh * sig
        results.append(gate_act * (consts.linear_beta * up_tanh))
    return results


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
    "SituV2Consts",
    "batched_silu_swiglu",
    "batched_situv2",
    "fclamp_f32",
    "fmin_f32",
    "fused_silu_swiglu_elem",
    "fused_situv2_elem",
    "make_lds_copy_ops",
    "pipeline_fence",
    "situv2_consts",
    "workgroup_barrier",
]
