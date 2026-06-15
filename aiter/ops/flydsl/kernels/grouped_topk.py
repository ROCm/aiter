# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Grouped (group-limited) Top-K gating kernel (FlyDSL).

Selects the ``topk`` routing experts for each token under a group-limited
constraint, in a single parameterized kernel that covers both the non-biased
and the biased variants (the variant is a compile-time constant):

    grouped_topk(gating_output,   # [num_tokens, num_experts]
                 bias,            # [num_experts]  (used only when is_biased)
                 topk_weights,    # [num_tokens, topk]  (row stride may be > topk)
                 topk_ids,        # [num_tokens, topk]
                 num_expert_group,
                 topk_group,
                 need_renorm,
                 is_softmax = True,
                 routed_scaling_factor = 1.0)

Phases:

  1. Score the experts of a token. Non-biased: softmax over all experts, or
     per-expert sigmoid. Biased: per-expert sigmoid plus a per-expert
     ``bias``, forming the *selection* score ``sigmoid(gate) + bias``.
  2. Reduce each expert-group to a single score: the group max (non-biased),
     or the sum of the group's top-2 selection scores (biased).
  3. Keep only the ``topk_group`` highest-scoring groups; mask out the rest.
  4. Select the global ``topk`` experts from the surviving groups. The stored
     weight is the score (non-biased) or the de-biased ``sigmoid(gate)``
     (biased, i.e. the bias is subtracted back out).
  5. Optionally renormalize the selected weights and apply
     ``routed_scaling_factor``.

Launch mapping:
  - grid  = (num_tokens, 1, 1)   -- one workgroup per token
  - block = (WARP_SIZE, 1, 1)    -- a single wavefront per token

Because each token is handled by a single wavefront, all cross-thread
reductions are warp shuffles (no cross-wave LDS reduction needed). LDS holds
the per-expert scores and the per-group scores for Phases 1-2; the global
top-k (Phase 3) is computed in registers with warp argmaxes (kept
register-resident so there is no per-pass barrier / LDS round-trip).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import arith, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T

from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from flydsl._mlir import ir
from flydsl.expr import buffer_ops, gpu, rocdl


WARP_SIZE = 64
_NEG_INF = float("-inf")
_LOG2E = 1.4426950408889634


def _elem_type(dtype_str: str):
    if dtype_str == "f32":
        return T.f32
    if dtype_str == "f16":
        return T.f16
    if dtype_str == "bf16":
        return T.bf16
    raise ValueError(f"unsupported gating dtype: {dtype_str}")


def build_grouped_topk_module(
    num_experts: int,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    need_renorm: bool,
    is_softmax: bool = True,
    dtype_str: str = "bf16",
    routed_scaling_factor: float = 1.0,
    is_biased: bool = False,
):
    """Return a JIT launcher for the grouped-topk gating kernel.

    All of ``num_experts``/``num_expert_group``/``topk_group``/``topk``/
    ``need_renorm``/``is_softmax``/``dtype_str``/``routed_scaling_factor``/
    ``is_biased`` are compile-time constants (the kernel is specialized per
    configuration). Only the output row strides remain runtime scalars.

    When ``is_biased`` is set (used when a per-expert ``bias`` is present):

      * scoring is always per-expert sigmoid (``is_softmax`` is forced off);
      * the per-expert *selection* score is ``sigmoid(gate) + bias[e]``;
      * a group's score is the **sum of its top-2** selection scores
        (vs. the single max in the non-biased path);
      * the stored weight of a selected expert is the **de-biased** value
        ``sigmoid(gate)`` (the bias is subtracted back out), and renorm /
        ``routed_scaling_factor`` operate on those de-biased weights.

    The kernel signature always carries a ``bias`` tensor argument; in the
    non-biased build it is never dereferenced (the wrapper passes a small
    placeholder)."""
    E = int(num_experts)
    G = int(num_expert_group)
    TG = int(topk_group)
    K = int(topk)
    is_biased = bool(is_biased)
    # The biased routing path always scores with sigmoid.
    if is_biased:
        is_softmax = False
    assert E % G == 0, f"num_experts({E}) must be divisible by num_expert_group({G})"
    assert 1 <= TG <= G, f"topk_group({TG}) must be in [1, {G}]"
    assert K <= WARP_SIZE, f"topk({K}) must be <= warp size({WARP_SIZE})"
    assert G <= WARP_SIZE, f"num_expert_group({G}) must be <= warp size({WARP_SIZE})"

    EPG = E // G  # experts per group
    # Top-2 group reduction needs at least 2 experts per group; with a single
    # expert the group score collapses to that expert's score.
    use_group_top2 = is_biased and EPG >= 2
    NITER = (E + WARP_SIZE - 1) // WARP_SIZE

    gpu_arch = get_hip_arch()
    sym_tag = (
        f"grouped_topk_lds_e{E}_g{G}_tg{TG}_k{K}_"
        f"{dtype_str}_sm{int(is_softmax)}_rn{int(need_renorm)}_b{int(is_biased)}"
    )
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name=sym_tag)
    f32_bytes = 4
    scores_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = scores_off + E * f32_bytes
    group_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = group_off + max(G, 1) * f32_bytes

    @flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
    def grouped_topk_kernel(
        gating: fx.Tensor,  # [num_tokens, num_experts] (in_elem)
        bias: fx.Tensor,  # [num_experts] (in_elem); unused unless is_biased
        topk_weights: fx.Tensor,  # [num_tokens, stride_w] f32
        topk_ids: fx.Tensor,  # [num_tokens, stride_id] i32
        stride_w: fx.Int32,
        stride_id: fx.Int32,
    ):
        f32 = T.f32
        i32 = T.i32
        in_elem = _elem_type(dtype_str)

        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        tid_v = ArithValue(tid)
        bid_v = ArithValue(bid)

        neg_inf = ArithValue(arith.constant(_NEG_INF, type=f32))
        c0f = ArithValue(arith.constant(0.0, type=f32))
        c1f = ArithValue(arith.constant(1.0, type=f32))
        c0i = ArithValue(arith.constant(0, type=i32))

        # Materialize LDS views at kernel entry (outside all scf regions) so the
        # memref.view ops dominate every store/load inside dynamic if/for blocks.
        base_ptr = allocator.get_base()
        s_scores = SmemPtr(base_ptr, scores_off, f32, shape=(E,))
        s_group = SmemPtr(base_ptr, group_off, f32, shape=(max(G, 1),))
        s_scores.get()
        s_group.get()

        g_rsrc = buffer_ops.create_buffer_resource(gating, max_size=True)
        w_rsrc = buffer_ops.create_buffer_resource(topk_weights, max_size=True)
        id_rsrc = buffer_ops.create_buffer_resource(topk_ids, max_size=True)
        if is_biased:
            bias_rsrc = buffer_ops.create_buffer_resource(bias, max_size=True)

        def _load_bias(col_safe):
            raw = buffer_ops.buffer_load(
                bias_rsrc, ArithValue(col_safe), vec_width=1, dtype=in_elem
            )
            return raw if dtype_str == "f32" else ArithValue(raw).extf(f32)

        row_off = bid_v * E

        def _idx(v):
            return fx.Index(v)

        def _wave_max(v):
            w = ArithValue(v)
            for sh in [32, 16, 8, 4, 2, 1]:
                peer = ArithValue(w.shuffle_xor(fx.Int32(sh), fx.Int32(WARP_SIZE)))
                w = ArithValue(w.maximumf(peer))
            return w

        def _wave_sum(v):
            w = ArithValue(v)
            for sh in [32, 16, 8, 4, 2, 1]:
                peer = ArithValue(w.shuffle_xor(fx.Int32(sh), fx.Int32(WARP_SIZE)))
                w = ArithValue(w.addf(peer))
            return w

        def _wave_argmax(val, idx):
            w = ArithValue(val)
            wi = ArithValue(idx)
            for sh in [32, 16, 8, 4, 2, 1]:
                pv = ArithValue(w.shuffle_xor(fx.Int32(sh), fx.Int32(WARP_SIZE)))
                pi = ArithValue(wi.shuffle_xor(fx.Int32(sh), fx.Int32(WARP_SIZE)))
                take = pv > w
                w = ArithValue(arith.select(take, pv, w))
                wi = ArithValue(arith.select(take, pi, wi))
            return w, wi

        # ── Phase 1: per-expert scores into LDS ──────────────────────────────
        if is_softmax:
            thread_max = neg_inf
            for i in range_constexpr(NITER):
                col = tid_v + i * WARP_SIZE
                valid = col < E
                col_safe = ArithValue(arith.select(valid, col, c0i))
                raw = buffer_ops.buffer_load(
                    g_rsrc, ArithValue(row_off + col_safe), vec_width=1, dtype=in_elem
                )
                g = raw if dtype_str == "f32" else ArithValue(raw).extf(f32)
                g = ArithValue(g)
                s_scores.store(g, [_idx(col_safe)])
                masked = ArithValue(arith.select(valid, g, neg_inf))
                thread_max = ArithValue(thread_max.maximumf(masked))

            row_max = _wave_max(thread_max)

            thread_sum = c0f
            for i in range_constexpr(NITER):
                col = tid_v + i * WARP_SIZE
                valid = col < E
                col_safe = ArithValue(arith.select(valid, col, c0i))
                g = ArithValue(s_scores.load([_idx(col_safe)]))
                e = ArithValue(((g - row_max) * _LOG2E)).exp2()
                e = ArithValue(e)
                s_scores.store(e, [_idx(col_safe)])
                contrib = ArithValue(arith.select(valid, e, c0f))
                thread_sum = ArithValue(thread_sum + contrib)

            row_sum = _wave_sum(thread_sum)
            inv_sum = ArithValue(c1f / row_sum)
            for i in range_constexpr(NITER):
                col = ArithValue(tid_v + i * WARP_SIZE)
                # NOTE: the AST rewriter only turns an ``if`` into scf.if when the
                # condition syntactically contains a call, hence ``fx.Int32(E)``.
                if col < fx.Int32(E):
                    g = ArithValue(s_scores.load([_idx(col)]))
                    s_scores.store(ArithValue(g * inv_sum), [_idx(col)])
        else:
            for i in range_constexpr(NITER):
                col = ArithValue(tid_v + i * WARP_SIZE)
                if col < fx.Int32(E):
                    raw = buffer_ops.buffer_load(
                        g_rsrc,
                        ArithValue(row_off + col),
                        vec_width=1,
                        dtype=in_elem,
                    )
                    g = raw if dtype_str == "f32" else ArithValue(raw).extf(f32)
                    g = ArithValue(g)
                    emu = ArithValue((-g * _LOG2E)).exp2()
                    sig = ArithValue(c1f / ArithValue(c1f + ArithValue(emu)))
                    if is_biased:
                        # Selection score = sigmoid(gate) + correction_bias[e].
                        # The bias is subtracted back out at store time so the
                        # final weight is the de-biased sigmoid value.
                        sig = ArithValue(sig + ArithValue(_load_bias(col)))
                    s_scores.store(sig, [_idx(col)])

        gpu.barrier()

        # ── Phase 2: group reduction + group-limited masking ─────────────────
        if G > 1:
            # 2a. group score:
            #   * non-biased  -> max over the experts of the group
            #   * biased      -> sum of the group's top-2 expert scores
            if tid_v < fx.Int32(G):
                gstart = ArithValue(tid_v * EPG)
                if use_group_top2:
                    max1 = neg_inf
                    max2 = neg_inf
                    for j in range_constexpr(EPG):
                        v = ArithValue(s_scores.load([_idx(gstart + j)]))
                        is_gt1 = v > max1
                        cand2 = ArithValue(arith.select(is_gt1, max1, v))
                        max2 = ArithValue(max2.maximumf(cand2))
                        max1 = ArithValue(max1.maximumf(v))
                    s_group.store(ArithValue(max1 + max2), [_idx(ArithValue(tid_v))])
                else:
                    gmax = neg_inf
                    for j in range_constexpr(EPG):
                        v = ArithValue(s_scores.load([_idx(gstart + j)]))
                        gmax = ArithValue(gmax.maximumf(v))
                    s_group.store(gmax, [_idx(ArithValue(tid_v))])
            gpu.barrier()

            # 2b. select the topk_group highest groups (mark them with -inf)
            if tid_v == fx.Int32(0):
                for _k in range_constexpr(TG):
                    mv = neg_inf
                    mi = c0i
                    for g in range_constexpr(G):
                        gv = ArithValue(s_group.load([_idx(g)]))
                        cond = gv > mv
                        g_const = ArithValue(arith.constant(g, type=i32))
                        mi = ArithValue(arith.select(cond, g_const, mi))
                        mv = ArithValue(arith.select(cond, gv, mv))
                    s_group.store(neg_inf, [_idx(mi)])
            gpu.barrier()

            # 2c. zero-out experts whose group was NOT selected
            for i in range_constexpr(NITER):
                col = ArithValue(tid_v + i * WARP_SIZE)
                if col < fx.Int32(E):
                    gidx = ArithValue(col // EPG)
                    gv = ArithValue(s_group.load([_idx(gidx)]))
                    cur = ArithValue(s_scores.load([_idx(col)]))
                    not_sel = gv != neg_inf
                    newv = ArithValue(arith.select(not_sel, neg_inf, cur))
                    s_scores.store(newv, [_idx(col)])
            gpu.barrier()

        # ── Phase 3: select global top-k experts ─────────────────────────────
        # Each lane loads its NITER masked scores into registers once, then the
        # K selection passes run in registers: every pass takes a warp argmax
        # and masks the winning slot via a broadcast-index compare.
        reg = []
        for i in range_constexpr(NITER):
            col = tid_v + i * WARP_SIZE
            valid = col < E
            col_safe = ArithValue(arith.select(valid, col, c0i))
            v = ArithValue(s_scores.load([_idx(col_safe)]))
            v = ArithValue(arith.select(valid, v, neg_inf))
            reg.append(v)

        topk_value = c0f
        topk_index = c0i
        sum_renorm = c0f

        for k in range_constexpr(K):
            local_max = neg_inf
            local_idx = c0i
            for i in range_constexpr(NITER):
                col = ArithValue(tid_v + i * WARP_SIZE)
                cond = reg[i] > local_max
                local_idx = ArithValue(arith.select(cond, col, local_idx))
                local_max = ArithValue(arith.select(cond, reg[i], local_max))

            max_val, max_idx = _wave_argmax(local_max, local_idx)
            # Broadcast lane 0's winning index to ALL lanes so every lane masks
            # and records the SAME expert. Ties make max_idx non-uniform across
            # lanes (max_val is already uniform); without this, per-lane masking
            # could drop two tied experts in one round.
            max_idx = ArithValue(rocdl.readfirstlane(T.i32, max_idx))

            # Mask the winning register slot on whichever lane owns it.
            for i in range_constexpr(NITER):
                col = ArithValue(tid_v + i * WARP_SIZE)
                is_win = col == max_idx
                reg[i] = ArithValue(arith.select(is_win, neg_inf, reg[i]))

            # For the biased path the winner was ranked by (sigmoid + bias);
            # the stored weight and the renorm sum use the de-biased sigmoid
            # value (the bias is subtracted back out).
            if is_biased:
                sel_val = ArithValue(max_val - ArithValue(_load_bias(max_idx)))
            else:
                sel_val = max_val

            is_k = tid_v == fx.Int32(k)
            topk_value = ArithValue(arith.select(is_k, sel_val, topk_value))
            topk_index = ArithValue(arith.select(is_k, max_idx, topk_index))
            sum_renorm = ArithValue(sum_renorm + sel_val)

        # ── Phase 4: renorm + scale + store ──────────────────────────────────
        scale = ArithValue(arith.constant(float(routed_scaling_factor), type=f32))
        if need_renorm:
            factor = ArithValue(scale / sum_renorm)
        else:
            factor = scale

        if tid_v < fx.Int32(K):
            w_off = bid_v * ArithValue(stride_w) + tid_v
            id_off = bid_v * ArithValue(stride_id) + tid_v
            buffer_ops.buffer_store(
                ArithValue(topk_value * factor), w_rsrc, ArithValue(w_off)
            )
            buffer_ops.buffer_store(topk_index, id_rsrc, ArithValue(id_off))

    @flyc.jit
    def launch_grouped_topk(
        gating: fx.Tensor,
        bias: fx.Tensor,
        topk_weights: fx.Tensor,
        topk_ids: fx.Tensor,
        stride_w: fx.Int32,
        stride_id: fx.Int32,
        num_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        idx_tokens = arith.index_cast(T.index, num_tokens)
        launcher = grouped_topk_kernel(
            gating, bias, topk_weights, topk_ids, stride_w, stride_id
        )
        launcher.launch(
            grid=(idx_tokens, 1, 1),
            block=(WARP_SIZE, 1, 1),
            stream=stream,
        )

    return launch_grouped_topk
