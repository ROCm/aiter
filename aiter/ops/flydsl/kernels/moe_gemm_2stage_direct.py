"""Small-M direct MoE stage2 kernel.

This path is intentionally narrow: fp8 A2/W2, per-tensor/per-expert f32
scales, bf16 output.  It avoids sorted stage2 padding and global atomics by
assigning one workgroup to (token, N-tile) and reducing topk in-kernel.
"""

import functools
from contextlib import contextmanager

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T

from .mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    load_b_pack_k32,
    make_preshuffle_b_layout,
)


@contextmanager
def _if_then(if_op):
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


@contextmanager
def _if_else(if_op):
    with ir.InsertionPoint(if_op.else_block):
        try:
            yield if_op.else_block
        finally:
            blk = if_op.else_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _idx_to_llvm_ptr(idx_val, addr_space=1):
    idx_v = idx_val._value if hasattr(idx_val, "_value") else idx_val
    i64_v = arith.index_cast(T.i64, idx_v)
    i64_raw = i64_v._value if hasattr(i64_v, "_value") else i64_v
    return llvm.inttoptr(ir.Type.parse(f"!llvm.ptr<{addr_space}>"), i64_raw)


def _value(v):
    return v._value if hasattr(v, "_value") else v


def _s_nop(count=1):
    llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string=f"s_nop {count}",
        constraints="",
        has_side_effects=True,
    )


@functools.lru_cache(maxsize=256)
def compile_moe_gemm1_direct_smallm(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int = 16,
    tile_n: int = 64,
    tile_k: int = 64,
    in_dtype: str = "fp8",
    out_dtype: str = "bf16",
    a_scale_scalar: bool = True,
    w_scale_per_expert: bool = True,
    routes_per_block: int = 1,
    num_waves_override: int = 0,
    k_batch: int = 1,
    splitk_mode: str = "atomic",
):
    """Compile direct small-M stage1 for fp8/fp8 + silu(gate)*up.

    The grid is (N tile, token, topk slot). It bypasses expert padded
    `moe_sorting` for stage1 and writes [token, topk, inter_dim] directly.

    ``k_batch`` enables split-K parallelism along the model_dim axis (mirrors
    ``compile_moe_gemm1`` in moe_gemm_2stage.py).  Default 1 = no split.  When
    ``k_batch > 1``:
      * Each WG only processes ``model_dim / k_batch`` of the K reduction.
      * The Z grid is multiplied by ``k_batch`` (folded as
        ``z = topk_slot * k_batch + bz_kb`` so kb-siblings share the same A
        row + expert id; better L2 reuse).
      * The kernel writes **f32 pre-activation gate / up partials** (scaled by
        per-route * per-token * per-expert scales) instead of the final bf16
        ``silu(gate) * up`` — the host wrapper runs the silu+mul post-pass.
      * ``splitk_mode = "atomic"`` (default): partials are atomically added
        into a shared ``(tokens, topk, 2*inter_dim)`` f32 tmp buffer (gate at
        cols ``[0, inter_dim)``, up at ``[inter_dim, 2*inter_dim)``).
      * ``splitk_mode = "reduce"``: partials are plain-stored into a
        ``(k_batch, tokens, topk, 2*inter_dim)`` f32 tmp buffer (no atomics,
        no contention) and the host post-pass sums across the leading kb
        axis before silu+mul.  Trades ``kb*`` tmp memory for higher GEMM
        throughput at small M where atomic contention dominates.

    Constraints when ``k_batch > 1``:
      * ``model_dim % k_batch == 0`` AND ``(model_dim // k_batch) % tile_k == 0``
      * ``(model_dim // k_batch) // tile_k`` must be >= 1 (at least one K tile
        per WG -- this kernel uses a simple per-tile loop without 2-tile tail
        unrolling, so any non-zero K-tile count is fine, unlike the standard
        codegen which additionally requires an EVEN tile count).
    """

    if in_dtype != "fp8":
        raise ValueError(f"direct small-M stage1 supports only fp8, got {in_dtype!r}")
    if out_dtype != "bf16":
        raise ValueError(f"direct small-M stage1 supports only bf16, got {out_dtype!r}")
    if tile_m != 16:
        raise ValueError("direct MFMA stage1 currently requires tile_m=16")
    if tile_n < 16 or tile_n % 16 != 0:
        raise ValueError("direct MFMA stage1 requires tile_n divisible by 16")
    if tile_k % 64 != 0:
        raise ValueError("direct MFMA stage1 requires tile_k to be a multiple of 64")
    if model_dim % tile_k != 0:
        raise ValueError(f"model_dim={model_dim} must be divisible by tile_k={tile_k}")
    if inter_dim % tile_n != 0:
        raise ValueError(f"inter_dim={inter_dim} must be divisible by tile_n={tile_n}")
    if routes_per_block < 1 or topk % routes_per_block != 0:
        raise ValueError(
            f"routes_per_block={routes_per_block} must evenly divide topk={topk}"
        )

    num_waves = num_waves_override if num_waves_override > 0 else _ceil_div(tile_n, 64)
    if num_waves < 1:
        raise ValueError("direct MFMA stage1 requires at least one wave")
    if tile_n % num_waves != 0:
        raise ValueError(
            f"direct MFMA stage1 requires tile_n divisible by num_waves={num_waves}"
        )
    total_threads = num_waves * 64
    if total_threads > 1024:
        raise ValueError(
            f"direct MFMA stage1 block size exceeds HIP limit: {total_threads}"
        )
    n_per_wave = tile_n // num_waves
    if n_per_wave % 16 != 0:
        raise ValueError(
            f"direct MFMA stage1 requires n_per_wave={n_per_wave} divisible by 16"
        )
    num_acc_n = n_per_wave // 16
    block_threads = total_threads

    # ── Split-K validation ───────────────────────────────────────────────────
    _is_splitk = int(k_batch) > 1
    if _is_splitk:
        if int(model_dim) % int(k_batch) != 0:
            raise ValueError(
                f"compile_moe_gemm1_direct_smallm: model_dim={model_dim} not "
                f"divisible by k_batch={k_batch}"
            )
        _k_per_batch = int(model_dim) // int(k_batch)
        if _k_per_batch % int(tile_k) != 0:
            raise ValueError(
                f"compile_moe_gemm1_direct_smallm: K_per_batch={_k_per_batch} "
                f"(= model_dim / k_batch) not divisible by tile_k={tile_k}"
            )
        _tiles_per_batch = _k_per_batch // int(tile_k)
        if _tiles_per_batch < 1:
            raise ValueError(
                f"compile_moe_gemm1_direct_smallm: split-K leaves "
                f"tiles_per_batch={_tiles_per_batch} < 1; reduce k_batch."
            )
    else:
        _k_per_batch = int(model_dim)

    if str(splitk_mode) not in ("atomic", "reduce"):
        raise ValueError(
            f"compile_moe_gemm1_direct_smallm: splitk_mode must be 'atomic' or "
            f"'reduce', got {splitk_mode!r}"
        )
    _splitk_reduce = _is_splitk and str(splitk_mode) == "reduce"

    scale_tag = ("ass" if a_scale_scalar else "asr") + (
        "_wse" if w_scale_per_expert else "_wsn"
    )
    _kb_tag = f"_kb{int(k_batch)}" if _is_splitk else ""
    _skmode_tag = "_red" if _splitk_reduce else ""
    module_name = (
        f"direct_moe1_{in_dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}"
        f"_abi10_mfma_nolds_w{num_waves}_{scale_tag}"
        f"{'_rpb' + str(routes_per_block) if routes_per_block != 1 else ''}"
        f"{_kb_tag}{_skmode_tag}"
    )

    def out_elem():
        return T.bf16() if callable(T.bf16) else T.bf16

    def silu(x):
        t = x * (-1.4426950408889634)
        emu = rocdl.exp2(T.f32, t)
        den = 1.0 + emu
        sig = rocdl.rcp(T.f32, den)
        return x * sig

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def moe_gemm1_direct(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_topk_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
    ):
        tokens_in = arith.index_cast(T.index, i32_tokens_in)
        n_in = arith.index_cast(T.index, i32_n_in)
        k_in = arith.index_cast(T.index, i32_k_in)

        bx_n = gpu.block_id("x")
        by_tok = gpu.block_id("y")
        bz_packed = gpu.block_id("z")
        tx = gpu.thread_id("x")

        # Decode z = topk_slot * k_batch + bz_kb (kb-inner so kb-siblings
        # share the same A row + expert id, for L2 reuse).  When kb == 1
        # this collapses to bz_slot = bz_packed and bz_kb = 0 -- bz_kb is
        # only used to compute the per-WG K-slice base offset.
        if _is_splitk:
            bz_slot = bz_packed // fx.Index(int(k_batch))
            bz_kb = bz_packed % fx.Index(int(k_batch))
        else:
            bz_slot = bz_packed
            bz_kb = fx.Index(0)

        # ── Output buffer record-size ────────────────────────────────────
        # kb == 1            : bf16 (tokens, topk, inter_dim)        -> *2
        # kb >  1, atomic    : f32  (tokens, topk, 2*inter_dim)      -> *4 * 2
        # kb >  1, reduce    : f32  (kb, tokens, topk, 2*inter_dim)  -> *4 * 2 * kb
        if _is_splitk:
            _kb_factor = int(k_batch) if _splitk_reduce else 1
            out_nbytes = (
                tokens_in
                * fx.Index(topk)
                * fx.Index(inter_dim)
                * fx.Index(2 * 4 * _kb_factor)
            )
        else:
            out_nbytes = tokens_in * fx.Index(topk) * n_in * fx.Index(2)
        x_nbytes = tokens_in * k_in
        w_nbytes = fx.Index(experts) * fx.Index(2) * n_in * k_in
        scale_x_nbytes = (
            fx.Index(1) if a_scale_scalar else tokens_in
        ) * fx.Index(4)
        scale_w_nbytes = (
            fx.Index(experts)
            if w_scale_per_expert
            else fx.Index(experts) * fx.Index(2) * n_in
        ) * fx.Index(4)
        topk_nbytes = tokens_in * fx.Index(topk) * fx.Index(4)

        out_rsrc = buffer_ops.create_buffer_resource(
            arg_out, max_size=False, num_records_bytes=out_nbytes
        )
        x_rsrc = buffer_ops.create_buffer_resource(
            arg_x, max_size=False, num_records_bytes=x_nbytes
        )
        w_rsrc = buffer_ops.create_buffer_resource(
            arg_w, max_size=False, num_records_bytes=w_nbytes
        )
        sx_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_x, max_size=False, num_records_bytes=scale_x_nbytes
        )
        sw_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_w, max_size=False, num_records_bytes=scale_w_nbytes
        )
        tid_rsrc = buffer_ops.create_buffer_resource(
            arg_topk_ids, max_size=False, num_records_bytes=topk_nbytes
        )

        b_layout = make_preshuffle_b_layout(
            arith,
            c_n=arith.index(experts * inter_dim * 2),
            c_k=k_in,
            kpack_bytes=16,
            elem_bytes=1,
        )
        layout_b = b_layout.layout_b
        layout_tx_wave_lane = fx.make_layout((num_waves, 64), stride=(64, 1))
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
        layout_n_blk_intra = fx.make_layout(
            (experts * inter_dim * 2 // 16, 16), stride=(16, 1)
        )

        coord_wl = fx.idx2crd(tx, layout_tx_wave_lane)
        wave_id = fx.get(coord_wl, 0)
        lane_id = fx.get(coord_wl, 1)
        coord_l16 = fx.idx2crd(lane_id, layout_lane16)
        lane_div_16 = fx.get(coord_l16, 0)
        lane_mod_16 = fx.get(coord_l16, 1)

        by_n = bx_n * fx.Index(tile_n)
        n_tile_base = (wave_id % fx.Index(num_waves)) * fx.Index(n_per_wave)
        col_offset_base_bytes = lane_div_16 * fx.Index(16)
        acc_init = arith.constant_vector(0.0, T.f32x4)

        col_g_list = []
        gate_blk_list = []
        gate_intra_list = []
        up_blk_list = []
        up_intra_list = []
        for ni in range_constexpr(num_acc_n):
            col_g = by_n + n_tile_base + fx.Index(ni * 16) + lane_mod_16
            col_g_list.append(col_g)
            coord_gate = fx.idx2crd(col_g, layout_n_blk_intra)
            gate_blk_list.append(fx.get(coord_gate, 0))
            gate_intra_list.append(fx.get(coord_gate, 1))
            coord_up = fx.idx2crd(fx.Index(inter_dim) + col_g, layout_n_blk_intra)
            up_blk_list.append(fx.get(coord_up, 0))
            up_intra_list.append(fx.get(coord_up, 1))

        route_group_base = by_tok * fx.Index(topk) + bz_slot * fx.Index(routes_per_block)
        route_idx_list = []
        expert_i32_list = []
        expert_idx_list = []
        for rb in range_constexpr(routes_per_block):
            route_idx = route_group_base + fx.Index(rb)
            route_idx_list.append(route_idx)
            expert_i32 = buffer_ops.buffer_load(
                tid_rsrc, arith.index_cast(T.i32, route_idx), vec_width=1, dtype=T.i32
            )
            expert_i32_list.append(expert_i32)
            expert_idx_list.append(arith.index_cast(T.index, expert_i32))

        def load_a_packs_k64(base_k):
            idx_elem = (by_tok * k_in + base_k + col_offset_base_bytes) // fx.Index(4)
            loaded_a16 = buffer_copy_gmem16_dwordx4(
                buffer_ops,
                vector,
                elem_type=T.f8,
                idx_i32=idx_elem,
                rsrc=x_rsrc,
                vec_elems=16,
                elem_bytes=1,
            )
            a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
            a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
            a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
            return a0, a1

        def load_b_pair(expert_idx, base_k):
            expert_base_blk = expert_idx * fx.Index((2 * inter_dim) // 16)
            gate0 = []
            gate1 = []
            up0 = []
            up1 = []
            for ni in range_constexpr(num_acc_n):
                gate_blk = expert_base_blk + gate_blk_list[ni]
                up_blk = expert_base_blk + up_blk_list[ni]
                gate0.append(
                    load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=0,
                        n_blk=gate_blk,
                        n_intra=gate_intra_list[ni],
                        lane_div_16=lane_div_16,
                        elem_type=T.f8,
                        kpack_bytes=16,
                        elem_bytes=1,
                    )
                )
                gate1.append(
                    load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=1,
                        n_blk=gate_blk,
                        n_intra=gate_intra_list[ni],
                        lane_div_16=lane_div_16,
                        elem_type=T.f8,
                        kpack_bytes=16,
                        elem_bytes=1,
                    )
                )
                up0.append(
                    load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=0,
                        n_blk=up_blk,
                        n_intra=up_intra_list[ni],
                        lane_div_16=lane_div_16,
                        elem_type=T.f8,
                        kpack_bytes=16,
                        elem_bytes=1,
                    )
                )
                up1.append(
                    load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=1,
                        n_blk=up_blk,
                        n_intra=up_intra_list[ni],
                        lane_div_16=lane_div_16,
                        elem_type=T.f8,
                        kpack_bytes=16,
                        elem_bytes=1,
                    )
                )
            return gate0, gate1, up0, up1

        def mfma_k64(acc0, a0, a1, b0, b1):
            acc1 = rocdl.mfma_f32_16x16x32_fp8_fp8(
                T.f32x4, [a0, b0, acc0, 0, 0, 0]
            )
            return rocdl.mfma_f32_16x16x32_fp8_fp8(
                T.f32x4, [a1, b1, acc1, 0, 0, 0]
            )

        # Split-K: each WG processes only model_dim/k_batch of the K axis
        # starting at k_base_off = bz_kb * _k_per_batch.  When kb == 1 this
        # collapses to 0 and the loop walks the full model_dim as before.
        if _is_splitk:
            _k_base_off = bz_kb * fx.Index(int(_k_per_batch))
        else:
            _k_base_off = fx.Index(0)

        acc_gate = [[acc_init] * num_acc_n for _ in range_constexpr(routes_per_block)]
        acc_up = [[acc_init] * num_acc_n for _ in range_constexpr(routes_per_block)]
        for kt in range_constexpr(int(_k_per_batch) // tile_k):
            base_k = _k_base_off + fx.Index(kt * tile_k)
            for kk in range_constexpr(tile_k // 64):
                k_base = base_k + fx.Index(kk * 64)
                a0, a1 = load_a_packs_k64(k_base)
                for rb in range_constexpr(routes_per_block):
                    gate0, gate1, up0, up1 = load_b_pair(expert_idx_list[rb], k_base)
                    for ni in range_constexpr(num_acc_n):
                        acc_gate[rb][ni] = mfma_k64(
                            acc_gate[rb][ni], a0, a1, gate0[ni], gate1[ni]
                        )
                        acc_up[rb][ni] = mfma_k64(
                            acc_up[rb][ni], a0, a1, up0[ni], up1[ni]
                        )

        x_scale_idx = fx.Index(0) if a_scale_scalar else by_tok
        x_scale = buffer_ops.buffer_load(
            sx_rsrc,
            arith.index_cast(T.i32, x_scale_idx),
            vec_width=1,
            dtype=T.f32,
        )
        row0_lane = arith.cmpi(
            arith.CmpIPredicate.eq,
            arith.index_cast(T.i32, lane_div_16),
            arith.constant(0, type=T.i32),
        )

        # Pre-compute split-K kb-slice element base (only used when _is_splitk).
        # Element layout for both modes:
        #   atomic : (tokens, topk, 2*inter_dim) f32 -- one slice
        #   reduce : (k_batch, tokens, topk, 2*inter_dim) f32 -- kb slices
        # Within each row, gate is stored at cols [0, inter_dim) and up at
        # [inter_dim, 2*inter_dim) so the host silu_and_mul fold matches the
        # standard split-K layout in compile_moe_gemm1.
        if _is_splitk and _splitk_reduce:
            _slice_stride_idx = tokens_in * fx.Index(topk * 2 * inter_dim)
            _kb_base_idx = bz_kb * _slice_stride_idx
        else:
            _kb_base_idx = fx.Index(0)

        _if_row0 = scf.IfOp(row0_lane)
        with _if_then(_if_row0):
            for rb in range_constexpr(routes_per_block):
                if w_scale_per_expert:
                    w_scale = buffer_ops.buffer_load(
                        sw_rsrc,
                        expert_i32_list[rb],
                        vec_width=1,
                        dtype=T.f32,
                    )
                else:
                    w_scale = arith.constant(1.0, type=T.f32)
                route_scale = x_scale * w_scale
                for ni in range_constexpr(num_acc_n):
                    gate_v = vector.extract(
                        acc_gate[rb][ni], static_position=[0], dynamic_position=[]
                    ) * route_scale
                    up_v = vector.extract(
                        acc_up[rb][ni], static_position=[0], dynamic_position=[]
                    ) * route_scale
                    if not w_scale_per_expert:
                        sw_gate_idx = (
                            expert_idx_list[rb] * fx.Index(2 * inter_dim)
                            + col_g_list[ni]
                        )
                        sw_up_idx = sw_gate_idx + fx.Index(inter_dim)
                        sw_gate = buffer_ops.buffer_load(
                            sw_rsrc,
                            arith.index_cast(T.i32, sw_gate_idx),
                            vec_width=1,
                            dtype=T.f32,
                        )
                        sw_up = buffer_ops.buffer_load(
                            sw_rsrc,
                            arith.index_cast(T.i32, sw_up_idx),
                            vec_width=1,
                            dtype=T.f32,
                        )
                        gate_v = gate_v * sw_gate
                        up_v = up_v * sw_up
                    if _is_splitk:
                        # Pre-activation f32 partials.  No silu, no mul, no
                        # trunc -- the host post-pass runs silu_and_mul after
                        # the kb reduction (atomic-fadd or kb-axis sum).
                        row_base_idx = (
                            _kb_base_idx
                            + route_idx_list[rb] * fx.Index(2 * inter_dim)
                        )
                        idx_g_idx = row_base_idx + col_g_list[ni]
                        idx_u_idx = idx_g_idx + fx.Index(inter_dim)
                        if _splitk_reduce:
                            buffer_ops.buffer_store(
                                gate_v,
                                out_rsrc,
                                arith.index_cast(T.i32, idx_g_idx),
                            )
                            buffer_ops.buffer_store(
                                up_v,
                                out_rsrc,
                                arith.index_cast(T.i32, idx_u_idx),
                            )
                        else:
                            # atomic-fadd: byte offset = element idx * 4.
                            idx_g_i32 = arith.index_cast(T.i32, idx_g_idx)
                            idx_u_i32 = arith.index_cast(T.i32, idx_u_idx)
                            c4_i32 = arith.constant(4, type=T.i32)
                            zero_i32 = arith.constant(0, type=T.i32)
                            byte_g = idx_g_i32 * c4_i32
                            byte_u = idx_u_i32 * c4_i32
                            rocdl.raw_ptr_buffer_atomic_fadd(
                                gate_v, out_rsrc, byte_g,
                                zero_i32, zero_i32,
                            )
                            rocdl.raw_ptr_buffer_atomic_fadd(
                                up_v, out_rsrc, byte_u,
                                zero_i32, zero_i32,
                            )
                    else:
                        out_v = arith.trunc_f(out_elem(), silu(gate_v) * up_v)
                        out_idx = route_idx_list[rb] * n_in + col_g_list[ni]
                        buffer_ops.buffer_store(
                            out_v, out_rsrc, arith.index_cast(T.i32, out_idx)
                        )

    @flyc.jit
    def launch_moe_gemm1_direct(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_topk_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        stream: fx.Stream,
    ):
        gx = fx.Index(_ceil_div(inter_dim, tile_n))
        gy = arith.index_cast(T.index, i32_tokens_in)
        # Split-K folds kb into z: z = topk_slot * kb + bz_kb.  When kb == 1
        # this collapses to the original (topk // routes_per_block) z extent.
        gz = fx.Index((topk // routes_per_block) * int(k_batch))
        moe_gemm1_direct(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_topk_ids,
            i32_tokens_in,
            i32_n_in,
            i32_k_in,
        ).launch(
            grid=(gx, gy, gz),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch_moe_gemm1_direct.__name__ = module_name
    return launch_moe_gemm1_direct


@functools.lru_cache(maxsize=256)
def compile_moe_gemm2_direct_smallm(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int = 16,
    tile_n: int = 64,
    tile_k: int = 64,
    in_dtype: str = "fp8",
    out_dtype: str = "bf16",
    a_scale_scalar: bool = True,
    w_scale_per_expert: bool = True,
    split_reduce: bool = False,
):
    """Compile direct small-M stage2 using MFMA fragments.

    `tile_m` is a dummy 16-row MFMA tile. For each token/topk route we fill the
    tile with the same A row, multiply by that route's expert W2 tile, and keep
    one row of the fragment. This preserves the direct single-write contract
    while avoiding scalar fp8 decode/dot in the hot loop.
    """

    if in_dtype != "fp8":
        raise ValueError(f"direct small-M stage2 supports only fp8, got {in_dtype!r}")
    if out_dtype != "bf16":
        raise ValueError(f"direct small-M stage2 supports only bf16, got {out_dtype!r}")
    if tile_m < topk:
        raise ValueError(f"tile_m={tile_m} must cover topk={topk}")
    if inter_dim % tile_k != 0:
        raise ValueError(f"inter_dim={inter_dim} must be divisible by tile_k={tile_k}")
    if model_dim % tile_n != 0:
        raise ValueError(f"model_dim={model_dim} must be divisible by tile_n={tile_n}")

    if tile_m != 16:
        raise ValueError("direct MFMA stage2 currently requires tile_m=16")
    if tile_n % 32 != 0:
        raise ValueError("direct MFMA stage2 requires tile_n divisible by 32")
    if tile_k != 64:
        raise ValueError("direct MFMA stage2 currently requires tile_k=64")

    total_threads = (tile_n // 32) * 64
    if total_threads > 1024:
        raise ValueError(
            f"direct MFMA stage2 block size exceeds HIP limit: {total_threads}"
        )
    num_waves = total_threads // 64
    n_per_wave = tile_n // num_waves
    num_acc_n = n_per_wave // 16

    block_threads = total_threads
    scale_tag = ("ass" if a_scale_scalar else "asr") + (
        "_wse" if w_scale_per_expert else "_wsn"
    )
    module_name = (
        f"direct_moe2_{in_dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}"
        f"_abi8_mfma_nolds_{scale_tag}"
        f"{'_splitreduce' if split_reduce else ''}"
    )

    def out_elem():
        return T.bf16() if callable(T.bf16) else T.bf16

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def moe_gemm2_direct(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_topk_ids: fx.Tensor,
        arg_topk_weights: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
    ):
        tokens_in = arith.index_cast(T.index, i32_tokens_in)
        n_in = arith.index_cast(T.index, i32_n_in)
        k_in = arith.index_cast(T.index, i32_k_in)

        bx_n = gpu.block_id("x")
        by_tok = gpu.block_id("y")
        tx = gpu.thread_id("x")

        out_rows = tokens_in * (fx.Index(topk) if split_reduce else fx.Index(1))
        out_nbytes = out_rows * n_in * fx.Index(2)
        x_nbytes = tokens_in * fx.Index(topk) * k_in
        w_nbytes = fx.Index(experts) * n_in * k_in
        scale_x_nbytes = (
            fx.Index(1) if a_scale_scalar else tokens_in * fx.Index(topk)
        ) * fx.Index(4)
        scale_w_nbytes = (
            fx.Index(experts) if w_scale_per_expert else fx.Index(experts) * n_in
        ) * fx.Index(4)
        topk_nbytes = tokens_in * fx.Index(topk) * fx.Index(4)

        out_rsrc = buffer_ops.create_buffer_resource(
            arg_out, max_size=False, num_records_bytes=out_nbytes
        )
        x_rsrc = buffer_ops.create_buffer_resource(
            arg_x, max_size=False, num_records_bytes=x_nbytes
        )
        w_rsrc = buffer_ops.create_buffer_resource(
            arg_w, max_size=False, num_records_bytes=w_nbytes
        )
        sx_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_x, max_size=False, num_records_bytes=scale_x_nbytes
        )
        sw_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_w, max_size=False, num_records_bytes=scale_w_nbytes
        )
        tid_rsrc = buffer_ops.create_buffer_resource(
            arg_topk_ids, max_size=False, num_records_bytes=topk_nbytes
        )
        tw_rsrc = buffer_ops.create_buffer_resource(
            arg_topk_weights, max_size=False, num_records_bytes=topk_nbytes
        )
        b_layout = make_preshuffle_b_layout(
            arith,
            c_n=arith.index(experts * model_dim),
            c_k=k_in,
            kpack_bytes=16,
            elem_bytes=1,
        )
        layout_b = b_layout.layout_b
        layout_tx_wave_lane = fx.make_layout((num_waves, 64), stride=(64, 1))
        layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))
        layout_n_blk_intra = fx.make_layout(
            (experts * model_dim // 16, 16), stride=(16, 1)
        )

        coord_wl = fx.idx2crd(tx, layout_tx_wave_lane)
        wave_id = fx.get(coord_wl, 0)
        lane_id = fx.get(coord_wl, 1)
        coord_l16 = fx.idx2crd(lane_id, layout_lane16)
        lane_div_16 = fx.get(coord_l16, 0)
        lane_mod_16 = fx.get(coord_l16, 1)

        by_n = bx_n * fx.Index(tile_n)
        n_tile_base = (wave_id % fx.Index(num_waves)) * fx.Index(n_per_wave)
        col_offset_base_bytes = lane_div_16 * fx.Index(16)
        acc_init = arith.constant_vector(0.0, T.f32x4)

        col_g_list = []
        n_blk_list = []
        n_intra_list = []
        for ni in range_constexpr(num_acc_n):
            col_g = by_n + n_tile_base + fx.Index(ni * 16) + lane_mod_16
            col_g_list.append(col_g)
            coord_w = fx.idx2crd(col_g, layout_n_blk_intra)
            n_blk_list.append(fx.get(coord_w, 0))
            n_intra_list.append(fx.get(coord_w, 1))

        def load_a_packs_k64(route_idx, base_k):
            idx_elem = (route_idx * k_in + base_k + col_offset_base_bytes) // fx.Index(4)
            loaded_a16 = buffer_copy_gmem16_dwordx4(
                buffer_ops,
                vector,
                elem_type=T.f8,
                idx_i32=idx_elem,
                rsrc=x_rsrc,
                vec_elems=16,
                elem_bytes=1,
            )
            a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
            a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
            a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
            return a0, a1

        def load_b_tile(expert_idx, base_k):
            expert_off_idx = expert_idx * n_in
            packs0 = []
            packs1 = []
            for ni in range_constexpr(num_acc_n):
                row_w = expert_off_idx + col_g_list[ni]
                coord_w = fx.idx2crd(row_w, layout_n_blk_intra)
                n_blk = fx.get(coord_w, 0)
                n_intra = fx.get(coord_w, 1)
                packs0.append(
                    load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=0,
                        n_blk=n_blk,
                        n_intra=n_intra,
                        lane_div_16=lane_div_16,
                        elem_type=T.f8,
                        kpack_bytes=16,
                        elem_bytes=1,
                    )
                )
                packs1.append(
                    load_b_pack_k32(
                        buffer_ops,
                        arith,
                        vector,
                        arg_b=arg_w,
                        b_rsrc=w_rsrc,
                        layout_b=layout_b,
                        base_k=base_k,
                        ki_step=1,
                        n_blk=n_blk,
                        n_intra=n_intra,
                        lane_div_16=lane_div_16,
                        elem_type=T.f8,
                        kpack_bytes=16,
                        elem_bytes=1,
                    )
                )
            return packs0, packs1

        def mfma_k64(acc0, a0, a1, b0, b1):
            acc1 = rocdl.mfma_f32_16x16x32_fp8_fp8(
                T.f32x4, [a0, b0, acc0, 0, 0, 0]
            )
            return rocdl.mfma_f32_16x16x32_fp8_fp8(
                T.f32x4, [a1, b1, acc1, 0, 0, 0]
            )

        out_acc = [arith.constant(0.0, type=T.f32)] * num_acc_n
        route_base = by_tok * fx.Index(topk)
        route_slot = gpu.block_id("z") if split_reduce else fx.Index(0)
        row0_lane = arith.cmpi(
            arith.CmpIPredicate.eq,
            arith.index_cast(T.i32, lane_div_16),
            arith.constant(0, type=T.i32),
        )

        for slot in range_constexpr(1 if split_reduce else topk):
            route_idx = route_base + (route_slot if split_reduce else fx.Index(slot))
            route_i32 = arith.index_cast(T.i32, route_idx)
            expert_i32 = buffer_ops.buffer_load(
                tid_rsrc, route_i32, vec_width=1, dtype=T.i32
            )
            expert_idx = arith.index_cast(T.index, expert_i32)
            x_scale_idx = fx.Index(0) if a_scale_scalar else route_idx
            x_scale = buffer_ops.buffer_load(
                sx_rsrc,
                arith.index_cast(T.i32, x_scale_idx),
                vec_width=1,
                dtype=T.f32,
            )
            route_weight = buffer_ops.buffer_load(
                tw_rsrc, route_i32, vec_width=1, dtype=T.f32
            )
            route_scale = x_scale * route_weight
            if w_scale_per_expert:
                sw_expert = buffer_ops.buffer_load(
                    sw_rsrc,
                    expert_i32,
                    vec_width=1,
                    dtype=T.f32,
                )
                route_scale = route_scale * sw_expert
            acc_slot = [acc_init] * num_acc_n

            for kt in range_constexpr(inter_dim // tile_k):
                base_k = fx.Index(kt * tile_k)
                b0, b1 = load_b_tile(expert_idx, base_k)
                a0, a1 = load_a_packs_k64(route_idx, base_k)
                for ni in range_constexpr(num_acc_n):
                    acc_slot[ni] = mfma_k64(acc_slot[ni], a0, a1, b0[ni], b1[ni])

            for ni in range_constexpr(num_acc_n):
                v = vector.extract(
                    acc_slot[ni], static_position=[0], dynamic_position=[]
                )
                if w_scale_per_expert:
                    out_acc[ni] = out_acc[ni] + (v * route_scale)
                else:
                    sw_idx = expert_idx * n_in + col_g_list[ni]
                    sw = buffer_ops.buffer_load(
                        sw_rsrc,
                        arith.index_cast(T.i32, sw_idx),
                        vec_width=1,
                        dtype=T.f32,
                    )
                    out_acc[ni] = out_acc[ni] + (v * route_scale * sw)

        _if_row0 = scf.IfOp(row0_lane)
        with _if_then(_if_row0):
            for ni in range_constexpr(num_acc_n):
                out_v = arith.trunc_f(out_elem(), out_acc[ni])
                out_idx = (
                    route_idx * n_in + col_g_list[ni]
                    if split_reduce
                    else by_tok * n_in + col_g_list[ni]
                )
                out_i32 = arith.index_cast(T.i32, out_idx)
                buffer_ops.buffer_store(out_v, out_rsrc, out_i32)

    if split_reduce:
        reduce_block_threads = 256
        reduce_vec_elems = 8
        reduce_tile_cols = reduce_block_threads * reduce_vec_elems

        @flyc.kernel(known_block_size=[reduce_block_threads, 1, 1])
        def moe_topk_reduce_direct(
            arg_tmp: fx.Tensor,
            arg_final: fx.Tensor,
            i32_tokens_in: fx.Int32,
        ):
            from flydsl._mlir.dialects import fly as _fly

            tokens_in = arith.index_cast(T.index, i32_tokens_in)
            token = gpu.block_id("x")
            tile = gpu.block_id("y")
            tid = gpu.thread_id("x")
            vec_i32 = T.vec(reduce_vec_elems // 2, T.i32)
            vec_bf16 = T.vec(reduce_vec_elems, T.bf16)
            vec_f32 = T.vec(reduce_vec_elems, T.f32)

            ptr_ty = ir.Type.parse("!llvm.ptr")
            tmp_base_ptr = _fly.extract_aligned_pointer_as_index(ptr_ty, arg_tmp)
            final_base_ptr = _fly.extract_aligned_pointer_as_index(ptr_ty, arg_final)
            tmp_base_idx = arith.index_cast(
                T.index, llvm.ptrtoint(T.i64, tmp_base_ptr)
            )
            final_base_idx = arith.index_cast(
                T.index, llvm.ptrtoint(T.i64, final_base_ptr)
            )

            col_base = (
                tile * fx.Index(reduce_tile_cols) + tid * fx.Index(reduce_vec_elems)
            )
            # The tuned small-M shapes use full 8-column vectors; keep a scalar tail
            # only for robustness on other model_dim values.
            col_ok = arith.cmpi(
                arith.CmpIPredicate.ult,
                arith.index_cast(T.i32, col_base),
                arith.constant(model_dim, type=T.i32),
            )
            full_ok = arith.cmpi(
                arith.CmpIPredicate.ule,
                arith.index_cast(T.i32, col_base + fx.Index(reduce_vec_elems)),
                arith.constant(model_dim, type=T.i32),
            )
            _if_col = scf.IfOp(col_ok)
            with _if_then(_if_col):
                _if_full = scf.IfOp(full_ok, has_else=True)
                with _if_then(_if_full):
                    acc = arith.constant_vector(0.0, vec_f32)
                    token_route_base = token * fx.Index(topk * model_dim)
                    for slot in range_constexpr(topk):
                        elem_idx = (
                            token_route_base + fx.Index(slot * model_dim) + col_base
                        )
                        byte_idx = tmp_base_idx + elem_idx * fx.Index(2)
                        raw = llvm.LoadOp(
                            vec_i32,
                            _idx_to_llvm_ptr(byte_idx),
                            alignment=16,
                        ).res
                        vals = vector.bitcast(vec_bf16, raw)
                        acc = acc + arith.extf(vec_f32, vals)
                    out_vec = arith.trunc_f(vec_bf16, acc)
                    out_elem_idx = token * fx.Index(model_dim) + col_base
                    out_byte_idx = final_base_idx + out_elem_idx * fx.Index(2)
                    llvm.StoreOp(
                        _value(out_vec),
                        _idx_to_llvm_ptr(out_byte_idx),
                        alignment=16,
                    )

                with _if_else(_if_full):
                    tmp_nbytes = tokens_in * fx.Index(topk * model_dim * 2)
                    final_nbytes = tokens_in * fx.Index(model_dim * 2)
                    tmp_rsrc = buffer_ops.create_buffer_resource(
                        arg_tmp, max_size=False, num_records_bytes=tmp_nbytes
                    )
                    final_rsrc = buffer_ops.create_buffer_resource(
                        arg_final, max_size=False, num_records_bytes=final_nbytes
                    )
                    token_route_base = token * fx.Index(topk * model_dim)
                    for lane in range_constexpr(reduce_vec_elems):
                        col = col_base + fx.Index(lane)
                        lane_ok = arith.cmpi(
                            arith.CmpIPredicate.ult,
                            arith.index_cast(T.i32, col),
                            arith.constant(model_dim, type=T.i32),
                        )
                        _if_lane = scf.IfOp(lane_ok)
                        with _if_then(_if_lane):
                            acc = arith.constant(0.0, type=T.f32)
                            for slot in range_constexpr(topk):
                                elem_idx = (
                                    token_route_base
                                    + fx.Index(slot * model_dim)
                                    + col
                                )
                                v = buffer_ops.buffer_load(
                                    tmp_rsrc,
                                    arith.index_cast(T.i32, elem_idx),
                                    vec_width=1,
                                    dtype=out_elem(),
                                )
                                acc = acc + arith.extf(T.f32, v)
                            out_v = arith.trunc_f(out_elem(), acc)
                            out_idx = token * fx.Index(model_dim) + col
                            buffer_ops.buffer_store(
                                out_v,
                                final_rsrc,
                                arith.index_cast(T.i32, out_idx),
                            )

        @flyc.jit
        def launch_moe_gemm2_direct(
            arg_final: fx.Tensor,
            arg_tmp: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
            arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor,
            arg_topk_ids: fx.Tensor,
            arg_topk_weights: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_n_in: fx.Int32,
            i32_k_in: fx.Int32,
            stream: fx.Stream,
        ):
            gx = fx.Index(_ceil_div(model_dim, tile_n))
            gy = arith.index_cast(T.index, i32_tokens_in)
            moe_gemm2_direct(
                arg_tmp,
                arg_x,
                arg_w,
                arg_scale_x,
                arg_scale_w,
                arg_topk_ids,
                arg_topk_weights,
                i32_tokens_in,
                i32_n_in,
                i32_k_in,
            ).launch(grid=(gx, gy, fx.Index(topk)), block=(block_threads, 1, 1), stream=stream)
            moe_topk_reduce_direct(arg_tmp, arg_final, i32_tokens_in).launch(
                grid=(
                    gy,
                    fx.Index(_ceil_div(model_dim, reduce_tile_cols)),
                    fx.Index(1),
                ),
                block=(reduce_block_threads, 1, 1),
                stream=stream,
            )
    else:

        @flyc.jit
        def launch_moe_gemm2_direct(
            arg_out: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
            arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor,
            arg_topk_ids: fx.Tensor,
            arg_topk_weights: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_n_in: fx.Int32,
            i32_k_in: fx.Int32,
            stream: fx.Stream,
        ):
            gx = fx.Index(_ceil_div(model_dim, tile_n))
            gy = arith.index_cast(T.index, i32_tokens_in)
            moe_gemm2_direct(
                arg_out,
                arg_x,
                arg_w,
                arg_scale_x,
                arg_scale_w,
                arg_topk_ids,
                arg_topk_weights,
                i32_tokens_in,
                i32_n_in,
                i32_k_in,
            ).launch(grid=(gx, gy, 1), block=(block_threads, 1, 1), stream=stream)

    launch_moe_gemm2_direct.__name__ = module_name
    return launch_moe_gemm2_direct


@functools.lru_cache(maxsize=64)
def compile_moe_route_fused_mfma_experimental(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int = 16,
    tile_n: int = 256,
    tile_k: int = 64,
    in_dtype: str = "fp8",
    out_dtype: str = "bf16",
    a_scale_scalar: bool = True,
    w_scale_per_expert: bool = True,
    split_reduce: bool = False,
):
    """Compile the MFMA-preserving route-fusion experiment backend.

    This is deliberately a narrow experimental entrypoint.  The current
    implementation reuses the direct stage2 MFMA kernel as the backend after
    the caller has produced fp8 A2 with the direct stage1 `_a2q` path.  Keeping
    this as a separate compile hook lets us benchmark and profile the
    MFMA-preserving route-fused API without touching production dispatch while
    leaving room to inline stage1 into this backend once the in-kernel A2 scale
    contract is solved.
    """

    return compile_moe_gemm2_direct_smallm(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        a_scale_scalar=a_scale_scalar,
        w_scale_per_expert=w_scale_per_expert,
        split_reduce=split_reduce,
    )


@functools.lru_cache(maxsize=16)
def compile_moe_fused_1kernel_smallm(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_n1: int = 16,
    tile_k1: int = 64,
    tile_n2: int = 256,
    tile_k2: int = 64,
    a_scale_scalar: bool = True,
    w_scale_per_expert: bool = True,
):
    """Single-kernel fused MoE: fp8 A1 -> stage1 MFMA -> silu -> LDS fp8 A2 -> stage2 MFMA -> bf16 out.

    Eliminates all inter-kernel data movement and dispatch overhead vs the
    5-kernel pipeline (A1-quant + stage1 + silu-A2q + stage2 + topk-reduce).

    Grid  : (ceil(model_dim / tile_n2), tokens, 1)
    Block : (tile_n2 // 32) * 64 threads

    Stage 1 (wave 0 only):
      For each route r in [0, topk):
        MFMA fp8xfp8 gate+up over all inter_dim N-tiles and model_dim K-tiles.
        Apply silu(gate)*up * x_scale * w1_scale_r  -> f32 A2.
        Track running abs-max per lane across N-tiles.
        Write 16 lane-maxima to LDS scratch; lane 0 reduces to per-route scale.
        Quantize f32 A2 -> fp8, store to LDS.

    Barrier (workgroup-wide)

    Stage 2 (all waves):
      Load fp8 A2 from LDS, MFMA fp8xfp8 with W2, accumulate weighted bf16 out.
    """
    if tile_n1 != 16:
        raise ValueError("fused 1-kernel requires tile_n1=16")
    if tile_k1 % 64 != 0:
        raise ValueError("fused 1-kernel requires tile_k1 divisible by 64")
    if tile_k2 % 64 != 0:
        raise ValueError("fused 1-kernel requires tile_k2 divisible by 64")
    if tile_n2 % 32 != 0:
        raise ValueError("fused 1-kernel requires tile_n2 divisible by 32")
    if model_dim % tile_k1 != 0:
        raise ValueError(f"model_dim={model_dim} not divisible by tile_k1={tile_k1}")
    if inter_dim % tile_n1 != 0:
        raise ValueError(f"inter_dim={inter_dim} not divisible by tile_n1={tile_n1}")
    if inter_dim % tile_k2 != 0:
        raise ValueError(f"inter_dim={inter_dim} not divisible by tile_k2={tile_k2}")
    if model_dim % tile_n2 != 0:
        raise ValueError(f"model_dim={model_dim} not divisible by tile_n2={tile_n2}")

    num_n1_tiles  = inter_dim // tile_n1
    num_k1_tiles  = model_dim // tile_k1
    num_k2_tiles  = inter_dim // tile_k2
    num_waves2    = tile_n2 // 32
    block_threads = num_waves2 * 64
    n_per_wave2   = tile_n2 // num_waves2
    num_acc_n2    = n_per_wave2 // 16

    _FP8_MAX = 240.0  # cvt_pk_fp8_f32 on CDNA3 uses E4M3FNUZ (max=240, not 448)

    # LDS layout (bytes, 16-byte aligned offsets)
    # _lds_a2f  : f32 A2 values (topk routes * inter_dim f32)  topk*inter_dim*4 bytes
    # _lds_sc   : f32 per-route A2 abs-max scale               topk*4 bytes
    # _lds_lm   : f32 lane-max scratch for scale reduction      topk*16*4 bytes
    # No fp8 LDS: stage2 loads f32 and packs on-the-fly via cvt_pk_fp8_f32
    _lds_a2f_b  = topk * inter_dim * 4   # f32 A2
    _lds_sc_b   = topk * 4               # f32 per-route abs-max scale
    _lds_lm_b   = topk * 16 * 4         # f32 lane-max scratch (16 lanes x topk)
    _off_a2f    = 0
    _off_sc     = _ceil_div(_off_a2f + _lds_a2f_b, 16) * 16
    _off_lm     = _ceil_div(_off_sc  + _lds_sc_b,  16) * 16
    _lds_total  = _ceil_div(_off_lm  + _lds_lm_b,  16) * 16

    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
    from flydsl.compiler.kernel_function import CompilationContext
    _alloc = SmemAllocator(None, arch="gfx942", global_sym_name="smem_fused1k")
    _alloc.ptr = _lds_total

    module_name = (
        f"fused_1k_moe_fp8_bf16"
        f"_t1x{tile_n1}x{tile_k1}_t2x{tile_n2}x{tile_k2}"
        f"_top{topk}_e{experts}"
    )

    def _out_elem():
        return T.bf16() if callable(T.bf16) else T.bf16

    def _silu(x):
        return x / (arith.constant(1.0, type=T.f32) + rocdl.exp2(T.f32, arith.constant(-1.4426950408889634, type=T.f32) * x))

    def _s_waitcnt_lgkm():
        llvm.InlineAsmOp(
            res=None, operands_=[],
            asm_string="s_waitcnt lgkmcnt(0)",
            constraints="", has_side_effects=True,
        )

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def moe_fused_1kernel(
        arg_out: fx.Tensor,
        arg_a1: fx.Tensor,
        arg_w1: fx.Tensor,
        arg_w2: fx.Tensor,
        arg_scale_a1: fx.Tensor,
        arg_scale_w1: fx.Tensor,
        arg_scale_w2: fx.Tensor,
        arg_topk_ids: fx.Tensor,
        arg_topk_weights: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k1_in: fx.Int32,
        i32_k2_in: fx.Int32,
    ):
        tokens_in = arith.index_cast(T.index, i32_tokens_in)
        n_in  = arith.index_cast(T.index, i32_n_in)
        k1_in = arith.index_cast(T.index, i32_k1_in)
        k2_in = arith.index_cast(T.index, i32_k2_in)

        bx_n2  = gpu.block_id("x")
        by_tok = gpu.block_id("y")
        tx     = gpu.thread_id("x")

        _bp          = _alloc.get_base()
        _lds_a2f_ptr = SmemPtr(_bp, _off_a2f, T.f32, shape=(topk * inter_dim,))
        _lds_sc_ptr  = SmemPtr(_bp, _off_sc,  T.f32, shape=(topk,))
        _lds_lm_ptr  = SmemPtr(_bp, _off_lm,  T.f32, shape=(topk * 16,))
        # Force memref.view creation at kernel body scope for SSA dominance
        _ = _lds_a2f_ptr.get()
        _ = _lds_sc_ptr.get()
        _ = _lds_lm_ptr.get()

        _out_nb  = tokens_in * n_in * fx.Index(2)
        _a1_nb   = tokens_in * k1_in
        _w1_nb   = fx.Index(experts * 2 * inter_dim) * k1_in
        _w2_nb   = fx.Index(experts) * n_in * k2_in
        _sa1_nb  = (fx.Index(1) if a_scale_scalar else tokens_in) * fx.Index(4)
        _sw_nb   = fx.Index(experts * 4)
        _tid_nb  = tokens_in * fx.Index(topk * 4)

        out_rsrc = buffer_ops.create_buffer_resource(arg_out,           max_size=False, num_records_bytes=_out_nb)
        a1_rsrc  = buffer_ops.create_buffer_resource(arg_a1,            max_size=False, num_records_bytes=_a1_nb)
        w1_rsrc  = buffer_ops.create_buffer_resource(arg_w1,            max_size=False, num_records_bytes=_w1_nb)
        w2_rsrc  = buffer_ops.create_buffer_resource(arg_w2,            max_size=False, num_records_bytes=_w2_nb)
        sa1_rsrc = buffer_ops.create_buffer_resource(arg_scale_a1,      max_size=False, num_records_bytes=_sa1_nb)
        sw1_rsrc = buffer_ops.create_buffer_resource(arg_scale_w1,      max_size=False, num_records_bytes=_sw_nb)
        sw2_rsrc = buffer_ops.create_buffer_resource(arg_scale_w2,      max_size=False, num_records_bytes=_sw_nb)
        tid_rsrc = buffer_ops.create_buffer_resource(arg_topk_ids,      max_size=False, num_records_bytes=_tid_nb)
        tw_rsrc  = buffer_ops.create_buffer_resource(arg_topk_weights,   max_size=False, num_records_bytes=_tid_nb)

        _ly_wl      = fx.make_layout((block_threads // 64, 64), stride=(64, 1))
        _ly_l16     = fx.make_layout((4, 16), stride=(16, 1))
        _crd_wl     = fx.idx2crd(tx, _ly_wl)
        wave_id     = fx.get(_crd_wl, 0)
        lane_id     = fx.get(_crd_wl, 1)
        _crd_l16    = fx.idx2crd(lane_id, _ly_l16)
        lane_div_16 = fx.get(_crd_l16, 0)
        lane_mod_16 = fx.get(_crd_l16, 1)

        is_wave0 = arith.cmpi(arith.CmpIPredicate.eq,
                               arith.index_cast(T.i32, wave_id),
                               arith.constant(0, type=T.i32))
        is_row0  = arith.cmpi(arith.CmpIPredicate.eq,
                               arith.index_cast(T.i32, lane_div_16),
                               arith.constant(0, type=T.i32))
        is_lane0 = arith.cmpi(arith.CmpIPredicate.eq,
                               arith.index_cast(T.i32, lane_id),
                               arith.constant(0, type=T.i32))

        col_off_bytes = lane_div_16 * fx.Index(16)

        _sa1_idx = fx.Index(0) if a_scale_scalar else by_tok
        x_scale  = buffer_ops.buffer_load(
            sa1_rsrc, arith.index_cast(T.i32, _sa1_idx), vec_width=1, dtype=T.f32)

        _b1_lay = make_preshuffle_b_layout(arith,
            c_n=arith.index(experts * inter_dim * 2), c_k=k1_in,
            kpack_bytes=16, elem_bytes=1)
        _lb1    = _b1_lay.layout_b
        _ly_n1b = fx.make_layout((experts * inter_dim * 2 // 16, 16), stride=(16, 1))

        _b2_lay = make_preshuffle_b_layout(arith,
            c_n=arith.index(experts * model_dim), c_k=k2_in,
            kpack_bytes=16, elem_bytes=1)
        _lb2    = _b2_lay.layout_b
        _ly_n2b = fx.make_layout((experts * model_dim // 16, 16), stride=(16, 1))

        # ============================================================
        # STAGE 1 — wave 0 only
        # ============================================================
        _if_w0 = scf.IfOp(is_wave0)
        with ir.InsertionPoint(_if_w0.then_block):
            _acc0 = arith.constant_vector(0.0, T.f32x4)

            def _ld_a1_k64(bk):
                _idx = (by_tok * k1_in + bk + col_off_bytes) // fx.Index(4)
                _v   = buffer_copy_gmem16_dwordx4(buffer_ops, vector,
                    elem_type=T.f8, idx_i32=_idx, rsrc=a1_rsrc,
                    vec_elems=16, elem_bytes=1)
                _i2  = vector.bitcast(T.i64x2, _v)
                return (vector.extract(_i2, static_position=[0], dynamic_position=[]),
                        vector.extract(_i2, static_position=[1], dynamic_position=[]))

            def _mfma(acc, a0, a1v, b0, b1):
                _t = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [a0, b0, acc, 0, 0, 0])
                return rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [a1v, b1, _t, 0, 0, 0])

            for _r in range_constexpr(topk):
                _ridx  = by_tok * fx.Index(topk) + fx.Index(_r)
                _ei32  = buffer_ops.buffer_load(
                    tid_rsrc, arith.index_cast(T.i32, _ridx), vec_width=1, dtype=T.i32)
                _eidx  = arith.index_cast(T.index, _ei32)
                _w1sc  = buffer_ops.buffer_load(sw1_rsrc, _ei32, vec_width=1, dtype=T.f32)
                _rsc   = x_scale * _w1sc
                _ebase = _eidx * fx.Index((2 * inter_dim) // 16)

                # --- Pass 1: MFMA (all 64 lanes) + silu + LDS store (row-0 lanes only) ---
                # MFMA ignores EXEC mask on CDNA3. All wave-0 lanes must participate
                # with valid fp8 A data so the accumulator does not see NaN/garbage.
                # Only row-0 lanes apply silu and write to LDS; extract(acc,[0])
                # holds the valid M=1 result for those lanes.
                _lane_max = arith.constant(0.0, type=T.f32)
                for _n1 in range_constexpr(num_n1_tiles):
                    _cg     = fx.Index(_n1 * 16) + lane_mod_16
                    _crdg   = fx.idx2crd(_cg, _ly_n1b)
                    _gblk   = _ebase + fx.get(_crdg, 0)
                    _gintra = fx.get(_crdg, 1)
                    _crdu   = fx.idx2crd(fx.Index(inter_dim) + _cg, _ly_n1b)
                    _ublk   = _ebase + fx.get(_crdu, 0)
                    _uintra = fx.get(_crdu, 1)
                    _ag = _acc0
                    _au = _acc0
                    for _kt in range_constexpr(num_k1_tiles):
                        for _kk in range_constexpr(tile_k1 // 64):
                            _kb  = fx.Index(_kt * tile_k1 + _kk * 64)
                            _a0, _a1v = _ld_a1_k64(_kb)
                            _bg0 = load_b_pack_k32(buffer_ops, arith, vector,
                                arg_b=arg_w1, b_rsrc=w1_rsrc, layout_b=_lb1,
                                base_k=_kb, ki_step=0, n_blk=_gblk, n_intra=_gintra,
                                lane_div_16=lane_div_16, elem_type=T.f8,
                                kpack_bytes=16, elem_bytes=1)
                            _bg1 = load_b_pack_k32(buffer_ops, arith, vector,
                                arg_b=arg_w1, b_rsrc=w1_rsrc, layout_b=_lb1,
                                base_k=_kb, ki_step=1, n_blk=_gblk, n_intra=_gintra,
                                lane_div_16=lane_div_16, elem_type=T.f8,
                                kpack_bytes=16, elem_bytes=1)
                            _bu0 = load_b_pack_k32(buffer_ops, arith, vector,
                                arg_b=arg_w1, b_rsrc=w1_rsrc, layout_b=_lb1,
                                base_k=_kb, ki_step=0, n_blk=_ublk, n_intra=_uintra,
                                lane_div_16=lane_div_16, elem_type=T.f8,
                                kpack_bytes=16, elem_bytes=1)
                            _bu1 = load_b_pack_k32(buffer_ops, arith, vector,
                                arg_b=arg_w1, b_rsrc=w1_rsrc, layout_b=_lb1,
                                base_k=_kb, ki_step=1, n_blk=_ublk, n_intra=_uintra,
                                lane_div_16=lane_div_16, elem_type=T.f8,
                                kpack_bytes=16, elem_bytes=1)
                            _ag = _mfma(_ag, _a0, _a1v, _bg0, _bg1)
                            _au = _mfma(_au, _a0, _a1v, _bu0, _bu1)
                    # All lanes: compute silu (row1-3 values finite but unused)
                    _gv  = vector.extract(_ag, static_position=[0], dynamic_position=[]) * _rsc
                    _uv  = vector.extract(_au, static_position=[0], dynamic_position=[]) * _rsc
                    _a2v = _silu(_gv) * _uv
                    _abs_a2v = llvm.call_intrinsic(T.f32, "llvm.fabs.f32", [_a2v], [], [])
                    _lane_max = arith.maximumf(_lane_max, _abs_a2v)
                    # Only row-0 lanes store to LDS (exec-masked ds_write)
                    _if_r0a = scf.IfOp(is_row0)
                    with ir.InsertionPoint(_if_r0a.then_block):
                        _a2f_i = fx.Index(_r * inter_dim + _n1 * 16) + lane_mod_16
                        _lds_a2f_ptr.store(_a2v, idxs=[_a2f_i])
                        scf.YieldOp([])
                # Store per-lane max to LDS scratch (row-0 lanes only)
                _if_r0b = scf.IfOp(is_row0)
                with ir.InsertionPoint(_if_r0b.then_block):
                    _lm_i = fx.Index(_r * 16) + lane_mod_16
                    _lds_lm_ptr.store(_lane_max, idxs=[_lm_i])
                    scf.YieldOp([])

                # Intra-wavefront fence: LDS writes from all row-0 lanes visible
                _s_waitcnt_lgkm()

                # Lane 0 reduces per-lane maxima -> per-route abs-max scale
                _if_l0 = scf.IfOp(is_lane0)
                with ir.InsertionPoint(_if_l0.then_block):
                    _gmax = arith.constant(0.0, type=T.f32)
                    for _li in range_constexpr(16):
                        _lm_val = _lds_lm_ptr.load(idxs=[fx.Index(_r * 16 + _li)])
                        _gmax = arith.maximumf(_gmax, _lm_val)
                    _fp8max = arith.constant(_FP8_MAX, type=T.f32)
                    _safe_max = arith.maximumf(_gmax, arith.constant(1e-12, type=T.f32))
                    _sc = _safe_max / _fp8max  # dequant scale: rcp used for packing in stage2
                    _lds_sc_ptr.store(_sc, idxs=[fx.Index(_r)])
                    scf.YieldOp([])

                _s_waitcnt_lgkm()
                # Note: no fp8 quantization pass here — stage2 loads f32 and
                # packs on-the-fly using rocdl.cvt_pk_fp8_f32.

            scf.YieldOp([])
        # end wave-0 if-block

        # ============================================================
        # Workgroup barrier: fp8 A2 in LDS visible to all waves
        # Use inline asm s_barrier (gpu.barrier() does not lower to s_barrier here)
        # ============================================================
        llvm.InlineAsmOp(
            res=None, operands_=[],
            asm_string="s_waitcnt lgkmcnt(0)\ns_barrier",
            constraints="", has_side_effects=True,
        )

        # ============================================================
        # STAGE 2 — all waves
        # ============================================================
        _by_n2   = bx_n2 * fx.Index(tile_n2)
        _n2base  = (wave_id % fx.Index(num_waves2)) * fx.Index(n_per_wave2)

        _col2s  = []
        _n2blks = []
        _n2intr = []
        for _ni in range_constexpr(num_acc_n2):
            _c2 = _by_n2 + _n2base + fx.Index(_ni * 16) + lane_mod_16
            _col2s.append(_c2)
            _cw = fx.idx2crd(_c2, _ly_n2b)
            _n2blks.append(fx.get(_cw, 0))
            _n2intr.append(fx.get(_cw, 1))

        _oacc  = [arith.constant(0.0, type=T.f32)] * num_acc_n2
        _ai2   = arith.constant_vector(0.0, T.f32x4)
        _rbase = by_tok * fx.Index(topk)

        for _sl in range_constexpr(topk):
            _ri2    = _rbase + fx.Index(_sl)
            _ri32_2 = arith.index_cast(T.i32, _ri2)
            _ei32_2 = buffer_ops.buffer_load(tid_rsrc, _ri32_2, vec_width=1, dtype=T.i32)
            _eidx_2 = arith.index_cast(T.index, _ei32_2)
            _tw_2   = buffer_ops.buffer_load(tw_rsrc,  _ri32_2, vec_width=1, dtype=T.f32)
            _a2sc2  = _lds_sc_ptr.load(idxs=[fx.Index(_sl)])
            _w2sc2  = buffer_ops.buffer_load(sw2_rsrc, _ei32_2, vec_width=1, dtype=T.f32)
            _csc    = _a2sc2 * _w2sc2 * _tw_2
            _eoff2  = _eidx_2 * n_in
            _asl    = [_ai2] * num_acc_n2

            for _kt2 in range_constexpr(num_k2_tiles):
                for _kk2 in range_constexpr(tile_k2 // 64):
                    _kb2 = fx.Index(_kt2 * tile_k2 + _kk2 * 64)

                    # Load f32 A2 from LDS, scale, pack 16xf32->16xfp8 via cvt_pk_fp8_f32
                    # _kb2 is the inter_dim element offset (already in f32-element units)
                    # lane_div_16 selects which 16-element stripe this lane-group handles
                    _a2sc_s2  = _lds_sc_ptr.load(idxs=[fx.Index(_sl)])
                    _a2_invsc = rocdl.rcp(T.f32, _a2sc_s2)
                    _a2f_base = fx.Index(_sl * inter_dim) + _kb2 + lane_div_16 * fx.Index(16)
                    _zero_i32 = arith.constant(0, type=T.i32)

                    # Pack f32[0..3] -> i32 _pk0a (4 fp8 bytes)
                    _fv0 = _lds_a2f_ptr.load(idxs=[_a2f_base + fx.Index(0)]) * _a2_invsc
                    _fv1 = _lds_a2f_ptr.load(idxs=[_a2f_base + fx.Index(1)]) * _a2_invsc
                    _pk0a = rocdl.cvt_pk_fp8_f32(T.i32, _fv0, _fv1, _zero_i32, 0)
                    _fv0 = _lds_a2f_ptr.load(idxs=[_a2f_base + fx.Index(2)]) * _a2_invsc
                    _fv1 = _lds_a2f_ptr.load(idxs=[_a2f_base + fx.Index(3)]) * _a2_invsc
                    _pk0a = rocdl.cvt_pk_fp8_f32(T.i32, _fv0, _fv1, _pk0a, 1)

                    # Pack f32[4..7] -> i32 _pk0b (4 fp8 bytes)
                    _fv0 = _lds_a2f_ptr.load(idxs=[_a2f_base + fx.Index(4)]) * _a2_invsc
                    _fv1 = _lds_a2f_ptr.load(idxs=[_a2f_base + fx.Index(5)]) * _a2_invsc
                    _pk0b = rocdl.cvt_pk_fp8_f32(T.i32, _fv0, _fv1, _zero_i32, 0)
                    _fv0 = _lds_a2f_ptr.load(idxs=[_a2f_base + fx.Index(6)]) * _a2_invsc
                    _fv1 = _lds_a2f_ptr.load(idxs=[_a2f_base + fx.Index(7)]) * _a2_invsc
                    _pk0b = rocdl.cvt_pk_fp8_f32(T.i32, _fv0, _fv1, _pk0b, 1)

                    # _ap0 = fp8[0..7] as i64
                    _pk0a_64 = arith.extui(T.i64, _pk0a)
                    _pk0b_64 = arith.extui(T.i64, _pk0b)
                    _ap0 = _pk0a_64 | (_pk0b_64 << arith.constant(32, type=T.i64))

                    # Pack f32[8..11] -> i32 _pk1a (4 fp8 bytes)
                    _fv0 = _lds_a2f_ptr.load(idxs=[_a2f_base + fx.Index(8)])  * _a2_invsc
                    _fv1 = _lds_a2f_ptr.load(idxs=[_a2f_base + fx.Index(9)])  * _a2_invsc
                    _pk1a = rocdl.cvt_pk_fp8_f32(T.i32, _fv0, _fv1, _zero_i32, 0)
                    _fv0 = _lds_a2f_ptr.load(idxs=[_a2f_base + fx.Index(10)]) * _a2_invsc
                    _fv1 = _lds_a2f_ptr.load(idxs=[_a2f_base + fx.Index(11)]) * _a2_invsc
                    _pk1a = rocdl.cvt_pk_fp8_f32(T.i32, _fv0, _fv1, _pk1a, 1)

                    # Pack f32[12..15] -> i32 _pk1b (4 fp8 bytes)
                    _fv0 = _lds_a2f_ptr.load(idxs=[_a2f_base + fx.Index(12)]) * _a2_invsc
                    _fv1 = _lds_a2f_ptr.load(idxs=[_a2f_base + fx.Index(13)]) * _a2_invsc
                    _pk1b = rocdl.cvt_pk_fp8_f32(T.i32, _fv0, _fv1, _zero_i32, 0)
                    _fv0 = _lds_a2f_ptr.load(idxs=[_a2f_base + fx.Index(14)]) * _a2_invsc
                    _fv1 = _lds_a2f_ptr.load(idxs=[_a2f_base + fx.Index(15)]) * _a2_invsc
                    _pk1b = rocdl.cvt_pk_fp8_f32(T.i32, _fv0, _fv1, _pk1b, 1)

                    # _ap1 = fp8[8..15] as i64
                    _pk1a_64 = arith.extui(T.i64, _pk1a)
                    _pk1b_64 = arith.extui(T.i64, _pk1b)
                    _ap1 = _pk1a_64 | (_pk1b_64 << arith.constant(32, type=T.i64))

                    for _ni in range_constexpr(num_acc_n2):
                        _rw2  = _eoff2 + _col2s[_ni]
                        _cw2n = fx.idx2crd(_rw2, _ly_n2b)
                        _b20  = load_b_pack_k32(buffer_ops, arith, vector,
                            arg_b=arg_w2, b_rsrc=w2_rsrc, layout_b=_lb2,
                            base_k=_kb2, ki_step=0,
                            n_blk=fx.get(_cw2n, 0), n_intra=fx.get(_cw2n, 1),
                            lane_div_16=lane_div_16, elem_type=T.f8,
                            kpack_bytes=16, elem_bytes=1)
                        _b21  = load_b_pack_k32(buffer_ops, arith, vector,
                            arg_b=arg_w2, b_rsrc=w2_rsrc, layout_b=_lb2,
                            base_k=_kb2, ki_step=1,
                            n_blk=fx.get(_cw2n, 0), n_intra=fx.get(_cw2n, 1),
                            lane_div_16=lane_div_16, elem_type=T.f8,
                            kpack_bytes=16, elem_bytes=1)
                        _t1      = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [_ap0, _b20, _asl[_ni], 0, 0, 0])
                        _asl[_ni] = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [_ap1, _b21, _t1, 0, 0, 0])

            for _ni in range_constexpr(num_acc_n2):
                _v = vector.extract(_asl[_ni], static_position=[0], dynamic_position=[])
                _oacc[_ni] = _oacc[_ni] + _v * _csc

        _if_out = scf.IfOp(is_row0)
        with ir.InsertionPoint(_if_out.then_block):
            for _ni in range_constexpr(num_acc_n2):
                _ov = arith.trunc_f(_out_elem(), _oacc[_ni])
                _oi = by_tok * n_in + _col2s[_ni]
                buffer_ops.buffer_store(_ov, out_rsrc, arith.index_cast(T.i32, _oi))
            scf.YieldOp([])


    @flyc.jit
    def launch_fused_1kernel(
        arg_out: fx.Tensor,
        arg_a1: fx.Tensor,
        arg_w1: fx.Tensor,
        arg_w2: fx.Tensor,
        arg_scale_a1: fx.Tensor,
        arg_scale_w1: fx.Tensor,
        arg_scale_w2: fx.Tensor,
        arg_topk_ids: fx.Tensor,
        arg_topk_weights: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k1_in: fx.Int32,
        i32_k2_in: fx.Int32,
        stream: fx.Stream,
    ):
        _alloc.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            _alloc.finalize()
        gx = fx.Index(_ceil_div(model_dim, tile_n2))
        gy = arith.index_cast(T.index, i32_tokens_in)
        moe_fused_1kernel(
            arg_out, arg_a1, arg_w1, arg_w2,
            arg_scale_a1, arg_scale_w1, arg_scale_w2,
            arg_topk_ids, arg_topk_weights,
            i32_tokens_in, i32_n_in, i32_k1_in, i32_k2_in,
        ).launch(
            grid=(gx, gy, fx.Index(1)),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch_fused_1kernel.__name__ = module_name
    return launch_fused_1kernel
