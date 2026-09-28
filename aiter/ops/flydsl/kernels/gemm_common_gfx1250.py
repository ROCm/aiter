"""Shared utilities for gfx1250 GEMM kernels (fp16 / mxfp4 / mxfp8)."""

from collections import namedtuple

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import as_ir_value, gpu, rocdl
from flydsl.expr.rocdl import cluster, tdm_ops
from flydsl.expr.typing import T

from .act import sigmoid_f32
from .kernels_common import LOG2E


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
        addr_i32 = fx.Int32(lds_base_idx) + fx.Int32(byte_offset)
        ptr = fx.inttoptr(ptr_ty, addr_i32)
        return fx.Tensor(fx.make_view(ptr, layout))

    def load(lds_base_idx, byte_offset):
        rmem = fx.make_rmem_tensor(layout, fx.Int32)
        fx.copy(atom, _view(lds_base_idx, byte_offset), rmem)
        return rmem.load()

    def store(lds_base_idx, byte_offset, data):
        rmem = fx.make_rmem_tensor(layout, fx.Int32)
        rmem.store(data)
        fx.copy(atom, rmem, _view(lds_base_idx, byte_offset))

    return load, store


def make_sgpr_opaque(val_i32):
    """Return ``val_i32`` unchanged but hidden from constant folding.

    A constexpr-unrolled region turns an LDS stage base (``base_ptr + s*PITCH``,
    where ``base_ptr`` is LDS offset 0) into a literal, so the backend folds it
    into every ds_load address and emits one v_add per load instead of one base
    register plus 16-bit ``offset:`` immediates -- the shape a rolled loop gets
    for free by keeping the stage in an SGPR. Routing the base through an
    "=s,s" asm gives the result no known definition, so the fold cannot happen.

    Emits no instruction.
    """
    op = llvm_dialect.InlineAsmOp(
        res=ir.IntegerType.get_signless(32),
        operands_=[as_ir_value(val_i32)],
        asm_string="",
        constraints="=s,s",
        has_side_effects=False,
        is_align_stack=False,
    )
    return op.res


def _raw_lds_ptr(lds_base_idx, byte_offset):
    """Materialize an LLVM LDS pointer from a pre-extracted byte base."""
    lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")
    total_byte = fx.Index(lds_base_idx) + fx.Index(byte_offset)
    return llvm_dialect.inttoptr(lds_ptr_ty, fx.Int32(total_byte).ir_value())


def lds_load_b128_raw(lds_base_idx, byte_offset):
    """Load 16 bytes from LDS using a pre-extracted base index (raw LLVM)."""
    ptr_val = _raw_lds_ptr(lds_base_idx, byte_offset)
    return llvm_dialect.load(
        ir.VectorType.get([4], ir.IntegerType.get_signless(32)), ptr_val
    )


def lds_load_b32_raw(lds_base_idx, byte_offset):
    """Load 4 bytes from LDS as ``i32`` using a pre-extracted base index."""
    ptr_val = _raw_lds_ptr(lds_base_idx, byte_offset)
    return llvm_dialect.load(ir.IntegerType.get_signless(32), ptr_val)


def lds_store_b128_raw(lds_base_idx, byte_offset, data):
    """Store 16 bytes to LDS using a pre-extracted base index (raw LLVM)."""
    ptr_val = _raw_lds_ptr(lds_base_idx, byte_offset)
    llvm_dialect.store(as_ir_value(data), ptr_val)


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


WGP_BARRIER_ID = -1


def pipeline_fence_signal(outstanding=0, use_cluster=False):
    """Signal the first half of a split pipeline fence."""
    tdm_ops.tensor_wait(outstanding)
    rocdl.s_barrier_signal(WGP_BARRIER_ID)
    if use_cluster:
        cluster.cluster_signal_once_per_wg()


def pipeline_fence_wait(use_cluster=False):
    """Wait for the second half of a split pipeline fence."""
    rocdl.s_barrier_wait(WGP_BARRIER_ID)
    if use_cluster:
        cluster.cluster_wait()


def fmin_f32(a, b):
    """Scalar f32 min (maps to v_min_num_f32)."""
    from flydsl.expr import arith

    return fx.Float32(arith.minnumf(as_ir_value(a), as_ir_value(b)))


def fclamp_f32(x, lo, hi):
    """Scalar f32 clamp via v_med3_num_f32."""
    return fx.Float32(
        rocdl.fmed3(T.f32, as_ir_value(x), as_ir_value(lo), as_ir_value(hi))
    )


def fused_silu_swiglu_elem(g, u, *, swiglu, limit_f32, neg_limit_f32):
    """One (gate, up) pair -> fused silu or swiglu scalar (gpt-oss clamp).

    Uses v_tanh_f32 (1 TRANS op) instead of exp2+rcp (2 TRANS ops):
        sigmoid(x) = 0.5*(1 + tanh(x/2))
    """
    _one = fx.Float32(1.0)
    _half = fx.Float32(0.5)
    g = fmin_f32(g, limit_f32)
    u = fclamp_f32(u, neg_limit_f32, limit_f32)
    if swiglu:
        _half_beta = fx.Float32(1.702 * 0.5)
        th = fx.Float32(rocdl.tanh(T.f32, as_ir_value(g * _half_beta)))
        sig = _half * (_one + th)
        return g * sig * (u + _one)
    th = fx.Float32(rocdl.tanh(T.f32, as_ir_value(g * _half)))
    sig = _half * (_one + th)
    return g * sig * u


def _tanh_f32(x, tanh_mul):
    """tanh(x) via the sigmoid identity tanh(z) = 2*sigmoid(2z) - 1.

    ``tanh_mul`` is the caller-hoisted ``-2*log2(e)/beta`` multiplier, so this
    evaluates ``2*rcp(1 + exp2(x * tanh_mul)) - 1`` for ``tanh(x/beta)`` in one
    exp2 + one rcp. Saturating rather than branchy: a large positive argument
    drives exp2 to +inf and rcp(+inf) to 0 (-> -1), a large negative one drives
    exp2 to 0 (-> +1), so no |x| fixup or sign select is needed.
    """
    _one = fx.Float32(1.0)
    _two = fx.Float32(2.0)
    exp_val = fx.Float32(rocdl.exp2(T.f32, as_ir_value(x * tanh_mul)))
    rcp_val = fx.Float32(rocdl.rcp(T.f32, _one + exp_val))
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
    neg_two_log2e = fx.Float32(-2.0 * LOG2E)
    return SituV2Consts(
        beta=beta,
        gate_tanh_mul=neg_two_log2e * fx.Float32(rocdl.rcp(T.f32, as_ir_value(beta))),
        linear_beta=linear_beta,
        up_tanh_mul=neg_two_log2e
        * fx.Float32(rocdl.rcp(T.f32, as_ir_value(linear_beta))),
    )


def fused_situv2_elem(g, u, *, consts):
    """One (gate, up) pair -> SiTUv2 (Kimi-K3 hidden_act="situ").

        beta * tanh(g/beta) * sigmoid(g) * linear_beta * tanh(u/linear_beta)

    ``consts`` comes from situv2_consts(). No clamp: SiTUv2 is bounded by
    construction, so the swiglu limit does not apply.
    """
    sig = sigmoid_f32(g)
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
    _one = fx.Float32(1.0)
    _two = fx.Float32(2.0)
    nlog2e = fx.Float32(-LOG2E)
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
        exp_vals.append(fx.Float32(rocdl.exp2(T.f32, as_ir_value(args[i]))))
    # Stage 2a: 1 + exp
    rocdl.sched_barrier(0)
    sum_vals = []
    for i in range_constexpr(3 * N):
        sum_vals.append(_one + exp_vals[i])
    # Stage 2b: rcp
    rocdl.sched_barrier(0)
    rcp_vals = []
    for i in range_constexpr(3 * N):
        rcp_vals.append(fx.Float32(rocdl.rcp(T.f32, sum_vals[i])))
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
    """Batched silu/swiglu with pipelined tanh for better TRANS utilisation.

    Uses v_tanh_f32 (1 TRANS op) instead of exp2+rcp (2 TRANS ops):
        sigmoid(x) = 0.5*(1 + tanh(x/2))

    Tanh is issued in groups of CHUNK; after each group a sched_barrier
    lets the compiler interleave the previous group's VALU consumers
    (add-one, mul-half, final products) with the next group's TRANS
    latency, avoiding a long TRANS-only stall.

    Args:
        pairs: list of (gate, up) f32 value pairs.
        swiglu: True for swiglu, False for silu.
        limit_f32, neg_limit_f32: clamp bounds.
        range_constexpr: the FlyDSL ``range_constexpr`` helper.

    Returns:
        list of activated f32 values, same length as *pairs*.
    """
    _one = fx.Float32(1.0)
    _half = fx.Float32(0.5)
    _half_scale = fx.Float32((1.702 * 0.5) if swiglu else 0.5)
    N = len(pairs)
    CHUNK = 4
    # Stage 1: clamp all pairs
    gs, us = [], []
    for i in range_constexpr(N):
        g = fmin_f32(pairs[i][0], limit_f32)
        u = fclamp_f32(pairs[i][1], neg_limit_f32, limit_f32)
        gs.append(g)
        us.append(u)
    rocdl.sched_barrier(0)
    # Stage 2+3 interleaved: issue CHUNK tanh, then consume previous CHUNK.
    tanh_vals = [None] * N
    results = [None] * N
    n_chunks = (N + CHUNK - 1) // CHUNK
    for c in range_constexpr(n_chunks):
        lo = c * CHUNK
        hi = min(lo + CHUNK, N)
        for i in range_constexpr(lo, hi):
            tanh_vals[i] = fx.Float32(
                rocdl.tanh(T.f32, as_ir_value(gs[i] * _half_scale))
            )
        rocdl.sched_barrier(0)
        prev_lo = (c - 1) * CHUNK if c > 0 else None
        prev_hi = lo if c > 0 else None
        if prev_lo is not None:
            for i in range_constexpr(prev_lo, prev_hi):
                sig = _half * (_one + tanh_vals[i])
                if swiglu:
                    results[i] = gs[i] * sig * (us[i] + _one)
                else:
                    results[i] = gs[i] * sig * us[i]
    # Drain last chunk's VALU
    drain_lo = (n_chunks - 1) * CHUNK
    for i in range_constexpr(drain_lo, N):
        sig = _half * (_one + tanh_vals[i])
        if swiglu:
            results[i] = gs[i] * sig * (us[i] + _one)
        else:
            results[i] = gs[i] * sig * us[i]
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
    "lds_load_b32_raw",
    "lds_load_b128_raw",
    "lds_store_b128_raw",
    "make_lds_copy_ops",
    "make_sgpr_opaque",
    "pipeline_fence",
    "pipeline_fence_signal",
    "pipeline_fence_wait",
    "situv2_consts",
    "workgroup_barrier",
]
