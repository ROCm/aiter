# SPDX-License-Identifier: MIT
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Software-pipeline stages: load, GEMM, softmax, store.

Part of the gfx950 dual-wave fp8 (e4m3fn) flash-attention kernel, migrated from
FlyDSL ``kernels/attention/flash_attn_utils.py`` and restricted to the symbols
the fp8 path reaches. The bf16/f16 dual-wave, the gfx942 generic path, paged KV,
and the bias/ALiBi helpers are not part of it and were left behind.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.fmha_gfx950.flash_attn_utils_primitives import (
    _LOG2E,
    NUM_XCD_GFX950,
    _anchor_scalar_f32,
    _anchor_v_o,
    _anchor_v_p,
    _apply_dualwave_causal_mask_pair,
    _attn_mask_vec2_imm,
    _buffer_load_128,
    _buffer_load_lds_128,
    _buffer_store_128,
    _causal_pair_thresholds,
    _cu_load,
    _ds_read_tr8_b64_imm,
    _exp2_score_slice,
    _pack_p_v8_slices,
    _read_exec_i64,
    _safe_l_inv,
    _scale_o_accs,
    _scale_sub_score_pair,
    _score_lists_to_vecs,
    _score_pair_max,
    _score_pair_sum,
    _score_pair_to_lists,
    _v_p_to_vec32,
    _v_pair_to_vec32,
    _v_vec32_to_p,
    _v_vec32_to_pair,
)


def _init_dualwave_thread_mapping(ctx):
    """Set block/wave/lane/head indices on a dualwave-style context.

    Shared verbatim by DualwaveKernelContext and DualwaveFp8KernelContext."""
    traits = ctx.traits
    batch_interleave_group = getattr(traits, "BATCH_INTERLEAVE_GROUP", 1)
    # Swizzled Head-first Mapping (arXiv:2511.02132): the grid is head-fast, so one
    # head's q-blocks scatter across all XCDs and each re-streams its K/V. Re-derive
    # (head, q_block) with head as the slow axis to keep them on one XCD. Bijective,
    # so output is bit-identical; split-K's third grid axis would not survive it.
    # Non-causal only: under a causal mask q-block i does work proportional to i, so
    # making q_block the fast axis clusters unequal work and costs 7% (measured).
    if const_expr(
        traits.XCD_SWIZZLE
        and not traits.SPLITK
        and not traits.CAUSAL
        and traits.NUM_HEADS_Q % NUM_XCD_GFX950 == 0
    ):
        num_q_blocks = fx.Index(gpu.grid_dim.y)
        linear_wg = fx.Index(gpu.block_idx.x) + fx.Index(gpu.block_idx.y) * fx.Index(
            traits.NUM_HEADS_Q
        )
        ctx.h_idx = linear_wg // num_q_blocks
        ctx.q_block_idx = linear_wg % num_q_blocks
    elif const_expr(batch_interleave_group > 1):
        linear_head_batch = fx.Index(gpu.block_idx.x)
        ctx.h_idx = linear_head_batch % traits.NUM_HEADS_Q
        ctx.batch_idx = (
            fx.Index(gpu.block_idx.z) * batch_interleave_group
            + linear_head_batch // traits.NUM_HEADS_Q
        )
        ctx.q_block_idx = fx.Index(gpu.block_idx.y)
    else:
        ctx.h_idx = fx.Index(gpu.block_idx.x)
        ctx.q_block_idx = fx.Index(gpu.block_idx.y)
    if const_expr(traits.SPLITK):
        ctx.bz_idx = fx.Index(gpu.block_idx.z)
        ctx.batch_idx = ctx.bz_idx // traits.NUM_KV_SPLITS
        ctx.split_idx = ctx.bz_idx % traits.NUM_KV_SPLITS
    elif const_expr(batch_interleave_group > 1):
        ctx.split_idx = None
    else:
        ctx.batch_idx = fx.Index(gpu.block_idx.z)
        ctx.split_idx = None
    ctx.tid = fx.Index(gpu.thread_idx.x)

    ctx.wave_id = ctx.tid // traits.WARP_SIZE
    ctx.lane = ctx.tid % traits.WARP_SIZE
    ctx.lane_mod_32 = ctx.lane % 32
    ctx.lane_div_32 = ctx.lane // 32

    _tid_i32 = fx.Int32(ctx.tid)
    _wave_id_uni_i32 = rocdl.readfirstlane(
        T.i32,
        (_tid_i32 // fx.Int32(traits.WARP_SIZE)).ir_value(),
    )
    # Two stagger groups, whatever the wave count.
    ctx.stagger_i32 = arith.divsi(
        _wave_id_uni_i32, as_mlir_value(fx.Int32(traits.NUM_WAVES // 2))
    )
    ctx.wave_id_uni = fx.Index(_wave_id_uni_i32)

    ctx.wave_q_offset = ctx.wave_id * traits.ROWS_PER_WAVE
    ctx.q_start = ctx.q_block_idx * traits.BLOCK_M

    ctx.h_kv_idx = ctx.h_idx % traits.NUM_HEADS_KV
    ctx.group_id = ctx.h_idx // traits.NUM_HEADS_KV
    ctx.q_head_idx = ctx.h_kv_idx * traits.GQA_GROUP_SIZE + ctx.group_id
    ctx.kv_head_idx = ctx.h_kv_idx


def _init_dualwave_q_row(ctx):
    """Set q_row / q_row_i32 / q_start_pos_i32 on a dualwave-style context."""
    traits = ctx.traits
    ctx.q_row_in_block = ctx.wave_q_offset + ctx.lane_mod_32
    ctx.q_start_pos_i32 = fx.Int32(ctx.q_start + ctx.wave_id_uni * traits.ROWS_PER_WAVE)
    ctx.q_row = ctx.q_start + ctx.q_row_in_block
    ctx.q_row_i32 = fx.Int32(ctx.q_row)


class DualwaveFp8KernelContext:
    """Shared per-kernel state for the gfx950 dualwave fp8 attention helpers.

    Mirrors ``DualwaveKernelContext`` but for the fp8 single path: raw fp8 Q/K/V
    (i8 buffer views), per-tensor Q/K/V descale scalars applied to the fp32 logits,
    and a bf16 ``vt`` LDS scratch for HIPREC PV."""

    def __init__(
        self,
        traits_or_ctx,
        Q=None,
        K=None,
        V=None,
        O=None,
        DebugCounts=None,
        CuSeqQ=None,
        CuSeqKv=None,
        QDescale=None,
        KDescale=None,
        VDescale=None,
        seq_len=None,
        seq_len_kv=None,
        stride_q_n=None,
        stride_kv_n=None,
        head_dim_runtime=None,
    ):
        if isinstance(traits_or_ctx, DualwaveFp8KernelContext):
            self.__dict__.update(traits_or_ctx.__dict__)
            self.ctx_ref = getattr(traits_or_ctx, "ctx_ref", traits_or_ctx)
            return
        self.ctx_ref = self
        self.traits = traits_or_ctx
        self.Q = Q
        self.K = K
        self.V = V
        self.O = O
        self.DebugCounts = DebugCounts
        self.CuSeqQ = CuSeqQ
        self.CuSeqKv = CuSeqKv
        self.QDescale = QDescale
        self.KDescale = KDescale
        self.VDescale = VDescale
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.stride_q_n = stride_q_n
        self.stride_kv_n = stride_kv_n
        self.head_dim_runtime = head_dim_runtime

    def init_types_and_constants(self):
        traits = self.traits
        self.elem_dtype = fx.Float8E4M3FN
        self.fm_fast = fx.arith.FastMathFlags.fast
        self.v4i32_type = Vec.make_type(4, fx.Int32)
        self.v4f16_type = Vec.make_type(4, self.elem_dtype)
        self.v16f32_type = Vec.make_type(16, fx.Float32)
        self.v2i32_type = Vec.make_type(2, fx.Int32)
        self.p_elem = fx.BFloat16
        self.v4bf16_type = Vec.make_type(4, fx.BFloat16)
        self.NUM_DMA_K = len(traits.K_BAND_CHUNK)
        self.c_neg_inf = fx.Float32(float("-inf"))
        self.c_neg_floor = fx.Float32(-3.0e38)
        self.c_zero_f = fx.Float32(0.0)
        self.c_rescale_thr_f = fx.Float32(traits.DUALWAVE_SWP_RESCALE_THRESHOLD)
        self.c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)

    def init_runtime_indices(self):
        traits = self.traits
        self.seq_len_v = fx.Index(self.seq_len)
        self.seq_len_kv_v = fx.Index(self.seq_len_kv)
        self.stride_q_n_v = fx.Index(self.stride_q_n)
        self.stride_kv_n_v = fx.Index(self.stride_kv_n)
        if traits.HEAD_DIM_V == traits.HEAD_DIM:
            self.stride_v_n_v = self.stride_kv_n_v
            self.stride_o_n_v = self.stride_q_n_v
        else:
            self.stride_v_n_v = traits.DEFAULT_STRIDE_V_N
            self.stride_o_n_v = traits.DEFAULT_STRIDE_O_N

    def init_causal_lpt_order(self):
        """Issue causal q-blocks longest-first by reversing the q-block grid axis.

        Causal work per q-block grows with the block index and workgroups dispatch in
        flattened-id order, so the natural order issues the heaviest block last and the
        makespan carries its tail. Must run after init_thread_mapping and before
        init_sequence_lengths / init_tile_bounds / init_q_row read q_start.
        """
        traits = self.traits
        num_q_blocks = (self.seq_len_v + traits.BLOCK_M - 1) // traits.BLOCK_M
        self.q_block_idx = num_q_blocks - 1 - self.q_block_idx
        self.q_start = self.q_block_idx * traits.BLOCK_M

    def init_lds(self, shared_storage):
        lds = fx.SharedAllocator().allocate(shared_storage).peek()
        self.lds = lds
        self.lds_kv_base_idx = fx.Index(fx.ptrtoint(lds.kv.ptr))
        self.lds_kv_base_ptr = lds.kv.ptr.llvm_ptr
        self.lds_vt_base_idx = fx.Index(fx.ptrtoint(lds.vt.ptr))
        self.lds_vt_base_ptr = lds.vt.ptr.llvm_ptr
        self.lds_q_base_idx = fx.Index(fx.ptrtoint(lds.q.ptr))
        self.lds_q_base_ptr = lds.q.ptr.llvm_ptr

    def init_thread_mapping(self):
        _init_dualwave_thread_mapping(self)

    def init_dma_thread_offsets(self):
        # Emitted after descriptors/atoms (matching the original schedule) so the
        # d_bucket ``v_and`` lands at the same ISA position.
        traits = self.traits
        self.lane_in_warp = self.tid % traits.WARP_SIZE
        self.n_in_warp = self.lane_in_warp // traits.LANE_SPLIT_KV
        self.d_bucket = self.lane_in_warp % traits.LANE_SPLIT_KV

    def init_sequence_lengths(self):
        traits = self.traits
        if const_expr(traits.VARLEN):
            _cuq_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(self.CuSeqQ), fx.make_layout(1, 1)
            )
            _cuk_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(self.CuSeqKv), fx.make_layout(1, 1)
            )
            _cu_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            _cu_v1i32 = Vec.make_type(1, fx.Int32)

            self.q_tok_base = _cu_load(_cuq_div, self.batch_idx, _cu_atom, _cu_v1i32)
            self.q_tok_end = _cu_load(_cuq_div, self.batch_idx + 1, _cu_atom, _cu_v1i32)
            self.kv_tok_base = _cu_load(_cuk_div, self.batch_idx, _cu_atom, _cu_v1i32)
            self.kv_tok_end = _cu_load(
                _cuk_div, self.batch_idx + 1, _cu_atom, _cu_v1i32
            )
            self.seqlen_q_v = self.q_tok_end - self.q_tok_base
            self.seqlen_kv_v = self.kv_tok_end - self.kv_tok_base
            self.seqlen_kv_i32 = fx.Int32(self.seqlen_kv_v)
        else:
            self.q_tok_base = self.batch_idx * self.seq_len_v
            self.kv_tok_base = self.batch_idx * self.seq_len_kv_v
            self.q_tok_end = (self.batch_idx + 1) * self.seq_len_v
            self.kv_tok_end = (self.batch_idx + 1) * self.seq_len_kv_v
            self.seqlen_q_v = self.seq_len_v
            self.seqlen_kv_v = self.seq_len_kv_v
            self.seqlen_kv_i32 = self.seq_len_kv
        self.delta_i32 = fx.Int32(self.seqlen_kv_i32 - fx.Int32(self.seqlen_q_v))
        self.q_gmem_elem_offset = (
            self.q_tok_base + self.q_start
        ) * self.stride_q_n_v + self.q_head_idx * traits.HEAD_DIM
        self.kv_gmem_elem_offset = (
            self.kv_tok_base * self.stride_kv_n_v + self.kv_head_idx * traits.HEAD_DIM
        )
        self.v_gmem_elem_offset = (
            self.kv_tok_base * self.stride_v_n_v + self.kv_head_idx * traits.HEAD_DIM_V
        )

    def init_descriptors(self):
        traits = self.traits
        eb = traits.ELEM_BYTES
        q_nrec_bytes = as_mlir_value(self.q_tok_end * self.stride_q_n_v * eb)
        kv_nrec_bytes = as_mlir_value(self.kv_tok_end * self.stride_kv_n_v * eb)
        v_nrec_bytes = as_mlir_value(self.kv_tok_end * self.stride_v_n_v * eb)
        o_nrec_bytes = as_mlir_value(
            self.q_tok_end * self.stride_o_n_v * traits.OUT_ELEM_BYTES
        )

        def _make_buf_div(tensor, nrec_bytes):
            # fp8 Q/K/V buffer views are i8-typed so DMA and register loads share one
            # byte view.
            bt = fx.rocdl.make_buffer_tensor(tensor, num_records_bytes=nrec_bytes)
            it = fx.get_iter(bt)
            i8_ptr_ty = fx.PointerType.get(
                elem_ty=fx.Int8.ir_type,
                address_space=fx.PointerType(it.type).address_space,
                alignment=fx.PointerType(it.type).alignment,
            )
            bt = fx.Tensor(
                fx.make_view(fx.recast_iter(i8_ptr_ty, it), fx.get_layout(bt))
            )
            return fx.logical_divide(bt, fx.make_layout(1, 1))

        self.q_div = _make_buf_div(self.Q, q_nrec_bytes)
        self.k_div = _make_buf_div(self.K, kv_nrec_bytes)
        self.v_div = _make_buf_div(self.V, v_nrec_bytes)
        self.o_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(self.O, num_records_bytes=o_nrec_bytes),
            fx.make_layout(1, 1),
        )

    def init_atoms_and_lds_ptrs(self):
        traits = self.traits
        self.load_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.load_atom_64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)
        self.store_atom_64 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)
        self.store_atom_128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        self.o_store_reg = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
        self.o_store_reg_128 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        # fp8 global->LDS DMA uses i8 destination typing; K/V LDS reads are byte-addressed.
        self.lds_ptr_ty = fx.PointerType.get(fx.Int8.ir_type, 2, traits.DMA_BYTES)

    def init_descale(self):
        def _load_scale_scalar(tensor):
            _div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(tensor), fx.make_layout(1, 1)
            )
            _atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            _v = fly.copy_atom_call_ssa(
                [Vec.make_type(1, fx.Float32)],
                _atom,
                fx.slice(_div, (None, fx.Int32(0))),
            )
            return fx.Float32(Vec(_v, (1,), fx.Float32)[0])

        head_dim_f32 = fx.Float32(fx.Int32(self.head_dim_runtime))
        c_log2e_f = fx.Float32(_LOG2E)
        c_sm_scale_log2e = fx.rsqrt(head_dim_f32, fastmath=self.fm_fast) * c_log2e_f
        _qd = _load_scale_scalar(self.QDescale)
        _kd = _load_scale_scalar(self.KDescale)
        self.vd_fp8 = _load_scale_scalar(self.VDescale)
        # fp8 feeds raw Q/K into the MFMA, so q/k descale * softmax scale multiplies
        # the fp32 logits after QK.
        self.c_logit_scale = c_sm_scale_log2e * (_qd * _kd)

    def init_tile_bounds(self):
        traits = self.traits
        kv_tile_size = traits.BLOCK_N
        num_kv_tiles = (self.seqlen_kv_v + kv_tile_size - 1) // kv_tile_size
        if const_expr(traits.CAUSAL):
            causal_end_raw_i32 = (
                fx.Int32(self.q_start + traits.BLOCK_M) + self.delta_i32
            )
            causal_end_i32 = fx.Int32(
                (causal_end_raw_i32 > fx.Int32(0)).select(
                    causal_end_raw_i32, fx.Int32(0)
                )
            )
            causal_num_tiles = (
                fx.Index(causal_end_i32) + kv_tile_size - 1
            ) // kv_tile_size
            max_num_tiles = fx.Index(
                (causal_num_tiles < num_kv_tiles).select(causal_num_tiles, num_kv_tiles)
            )
        else:
            causal_end_raw_i32 = None
            max_num_tiles = num_kv_tiles
        # Pipeline needs an EVEN tile count >= 4; extra tiles read 0 (num_records) and are masked.
        max_num_tiles = ((max_num_tiles + 1) // 2) * 2
        max_num_tiles = fx.Index((max_num_tiles < 4).select(4, max_num_tiles))
        self.max_num_tiles = max_num_tiles
        if const_expr(traits.SPLITK):
            chunk = (
                (
                    (max_num_tiles + (traits.NUM_KV_SPLITS - 1)) // traits.NUM_KV_SPLITS
                    + 1
                )
                // 2
                * 2
            )
            chunk = fx.Index((chunk < 6).select(6, chunk))
            split_t0 = self.split_idx * chunk
            split_t_end = split_t0 + chunk
            split_t_end = fx.Index(
                (split_t_end < max_num_tiles).select(split_t_end, max_num_tiles)
            )
            split_t_end = fx.Index(
                (max_num_tiles - split_t_end < 4).select(max_num_tiles, split_t_end)
            )
            self.split_nonempty = split_t0 + 4 <= max_num_tiles
        else:
            split_t0 = 0
            split_t_end = max_num_tiles
            self.split_nonempty = None

        if const_expr(traits.VARLEN or (traits.CAUSAL and traits.CROSS_SEQLEN)):
            active = None
            if const_expr(traits.VARLEN):
                active = self.q_start < self.seqlen_q_v
            if const_expr(traits.CAUSAL and traits.CROSS_SEQLEN):
                in_mask = causal_end_raw_i32 > fx.Int32(0)
                active = in_mask if active is None else (active & in_mask)
            split_t_end = fx.Index(active.select(split_t_end, split_t0))

        self.split_t0 = split_t0
        self.split_t_end = split_t_end

    def init_workspace_io(self):
        if const_expr(self.traits.SPLITK):
            self.ws_div = fx.logical_divide(
                fx.rocdl.make_buffer_tensor(self.DebugCounts), fx.make_layout(1, 1)
            )
            self.ws_store_atom_32 = fx.make_copy_atom(
                fx.rocdl.BufferCopy32b(), fx.Int32
            )
            self.ws_store_reg_32 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
            self.ws_store_reg_128 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)

    def ws_store_f32(self, f32_val, elem_index):
        pack = Vec.from_elements([fx.Float32(f32_val)], fx.Float32).bitcast(fx.Int32)
        fx.memref_store_vec(pack, self.ws_store_reg_32)
        fx.copy(
            self.ws_store_atom_32,
            self.ws_store_reg_32,
            fx.slice(self.ws_div, (None, fx.Int32(elem_index))),
        )

    def ws_store_quad_i32(self, dwords, elem_index):
        pack = Vec.from_elements([fx.Int32(v) for v in dwords], fx.Int32)
        fx.memref_store_vec(pack, self.ws_store_reg_128)
        fx.copy(
            self.store_atom_128,
            self.ws_store_reg_128,
            fx.slice(self.ws_div, (None, fx.Int32(elem_index))),
        )

    def init_q_row(self):
        _init_dualwave_q_row(self)

    def k_buf_base(self, buf_id):
        traits = self.traits
        if const_expr(isinstance(buf_id, int)):
            return traits.DUALWAVE_SWP_K_BUF_BASE[buf_id]
        return buf_id * traits.DUALWAVE_SWP_KV_PER_BUFFER

    def v_buf_base(self, buf_id):
        traits = self.traits
        if const_expr(isinstance(buf_id, int)):
            return traits.DUALWAVE_SWP_V_BUF_BASE[buf_id]
        return traits.SMEM_K_TILE_ELEMS + buf_id * traits.DUALWAVE_SWP_KV_PER_BUFFER

    def v_pair_to_vec32(self, v):
        return _v_pair_to_vec32(v)

    def v_vec32_to_pair(self, v):
        return _v_vec32_to_pair(v)

    def bf16_trunc_pack_v8(self, f32_vals):
        # HIPREC carries P/V as v8 bf16 regardless of the fp8 element dtype:
        # pack 8 f32 -> 4 cvt_pk_bf16 dwords.
        pairs = []
        for j in range_constexpr(4):
            pairs.append(rocdl.cvt_pk_bf16_f32(f32_vals[j * 2], f32_vals[j * 2 + 1]))
        return Vec.from_elements(pairs, fx.Int32).bitcast(fx.BFloat16).ir_value()

    def buffer_load_128(self, elem_index):
        return _buffer_load_128(
            elem_index, self.load_atom_128, self.q_div, self.v4i32_type
        )

    def buffer_load_lds_128(self, src_div, lds_byte_addr, src_elem, soffset_elems):
        _buffer_load_lds_128(
            src_div,
            lds_byte_addr,
            src_elem,
            soffset_elems,
            _dma_atom=self.dma_atom,
            _lds_ptr_ty=self.lds_ptr_ty,
        )

    def buffer_store_128(self, pack_i32_vec, elem_index):
        _buffer_store_128(
            pack_i32_vec,
            elem_index,
            self.o_store_reg_128,
            self.store_atom_128,
            self.o_div,
        )

    def global_idx_q(self, token_idx, col):
        return (
            (self.q_tok_base + token_idx) * self.stride_q_n_v
            + self.q_head_idx * self.traits.HEAD_DIM
            + col
        )

    def global_idx_o(self, token_idx, col):
        """Element index into O, which is HEAD_DIM_V wide (not HEAD_DIM)."""
        return (
            (self.q_tok_base + token_idx) * self.stride_o_n_v
            + self.q_head_idx * self.traits.HEAD_DIM_V
            + col
        )

    def read_i32x8_lds(self, base_ptr, byte_row):
        halves = []
        for h in range_constexpr(2):
            p = buffer_ops.get_element_ptr(
                base_ptr, byte_offset=fx.Int32(byte_row + h * 16), elem_type=T.i8
            )
            halves.append(
                Vec(llvm.LoadOp(Vec.make_type(4, fx.Int32), p, alignment=16).result)
            )
        return halves[0].shuffle(halves[1], [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()


class DualwaveFp8QLoader(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def stage_q_to_lds(self):
        traits = self.traits
        chunks_per_row = traits.HEAD_DIM // 16  # 16-byte DMA chunks per Q row
        total_chunks = (traits.BLOCK_M * traits.HEAD_DIM) // 16
        for p in range_constexpr(total_chunks // traits.BLOCK_SIZE):
            c = self.tid + (p * traits.BLOCK_SIZE)
            row = c // chunks_per_row
            dchunk = c % chunks_per_row
            src_elem = self.q_gmem_elem_offset + row * self.stride_q_n_v + dchunk * 16
            lds_addr = self.lds_q_base_idx + c * 16
            self.buffer_load_lds_128(self.q_div, lds_addr, src_elem, 0)

    def load_all_wide(self, q_row_in_block):
        traits = self.traits
        d_base = self.lane_div_32 * 32
        packs = []
        for ws in range_constexpr(traits.HEAD_DIM // 64):
            byte_row = q_row_in_block * traits.HEAD_DIM + (ws * 64) + d_base
            packs.append(self.read_i32x8_lds(self.lds_q_base_ptr, fx.Int32(byte_row)))
        return packs


class DualwaveFp8GemmHelper(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _mfma_acc_fp8_wide(self, a_i32x8, b_i32x8, c_v16):
        # Wide fp8 QK: mfma_scale (32x32x64) with unit E8M0 scales, i32x8 operands.
        return rocdl.mfma_scale_f32_32x32x64_f8f6f4(
            self.v16f32_type,
            [
                as_mlir_value(a_i32x8),
                as_mlir_value(b_i32x8),
                as_mlir_value(c_v16),
                0,
                0,
                0,
                as_mlir_value(fx.Int32(0x7F7F7F7F)),
                0,
                as_mlir_value(fx.Int32(0x7F7F7F7F)),
            ],
        )

    def _pack_fp8_i32x8(self, f32_vals):
        c0 = llvm.mlir_poison(T.i32)
        words = []
        for g in range_constexpr(8):
            base = g * 4
            w = rocdl.cvt_pk_fp8_f32(
                T.i32,
                as_mlir_value(f32_vals[base]),
                as_mlir_value(f32_vals[base + 1]),
                c0,
                0,
            )
            w = rocdl.cvt_pk_fp8_f32(
                T.i32,
                as_mlir_value(f32_vals[base + 2]),
                as_mlir_value(f32_vals[base + 3]),
                w,
                1,
            )
            words.append(fx.Int32(w))
        return Vec.from_elements(words, fx.Int32).ir_value()

    def _v_concat_i32x8(self, v_v, dc):
        words = []
        for ks in range_constexpr(4):
            v2 = Vec(
                llvm.bitcast(self.v2i32_type, as_mlir_value(v_v[ks][dc])),
                (2,),
                fx.Int32,
            )
            words.append(fx.Int32(v2[0]))
            words.append(fx.Int32(v2[1]))
        return Vec.from_elements(words, fx.Int32).ir_value()

    def _load_q_wide_lds(self):
        traits = self.traits
        q_row_in_block = self.ctx_ref.q_row_in_block
        d_base = self.lane_div_32 * 32
        packs = []
        for ws in range_constexpr(traits.HEAD_DIM // 64):
            byte_row = q_row_in_block * traits.HEAD_DIM + (ws * 64) + d_base
            packs.append(self.read_i32x8_lds(self.lds_q_base_ptr, fx.Int32(byte_row)))
        return packs

    def _load_q_wide_global(self):
        """Pull this lane's Q operands straight from global into VGPRs (head_dim > 128)."""
        traits = self.traits
        d_base = self.lane_div_32 * 32
        packs = []
        for ws in range_constexpr(traits.HEAD_DIM // 64):
            elem = self.global_idx_q(self.ctx_ref.q_row, (ws * 64) + d_base)
            lo = self.buffer_load_128(elem)
            hi = self.buffer_load_128(elem + 16)
            packs.append(Vec(lo).shuffle(Vec(hi), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value())
        return packs

    def load_q_wide(self):
        if const_expr(self.traits.QLDS):
            return self._load_q_wide_lds()
        return self._load_q_wide_global()

    def qk(self, v_k, q_wide=None):
        traits = self.traits
        k_lo, k_hi = v_k
        q_all_wide = self._load_q_wide_lds() if q_wide is None else q_wide
        v_s_lo = self.c_zero_v16f32
        v_s_hi = self.c_zero_v16f32
        for ws in range_constexpr(traits.HEAD_DIM // 64):
            q_w = q_all_wide[ws]
            v_s_lo = self._mfma_acc_fp8_wide(k_lo[ws], q_w, v_s_lo)
            v_s_hi = self._mfma_acc_fp8_wide(k_hi[ws], q_w, v_s_hi)
        n_ds = const_expr(traits.HEAD_DIM // 64 * 4)
        n_mfma = const_expr(traits.HEAD_DIM // 64 * 2)
        rocdl.sched_group_barrier(traits.SCHED_DS_READ_MASK, n_ds // 2, 12)
        rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, 1, 12)
        rocdl.sched_group_barrier(traits.SCHED_DS_READ_MASK, n_ds // 2, 12)
        rocdl.sched_group_barrier(traits.SCHED_MFMA_MASK, n_mfma - 1, 12)
        return (v_s_lo, v_s_hi)

    def cast_p_fp8_direct(self, v_p):
        lo_partial_list, hi_full = v_p
        f32 = []
        for pks in range_constexpr(self.traits.PV_K_STEPS):
            p_base = pks * 8
            f32 += [lo_partial_list[p_base + s] for s in range_constexpr(8)]
        for pks in range_constexpr(self.traits.PV_K_STEPS):
            p_base = pks * 8
            f32 += [hi_full[p_base + s] for s in range_constexpr(8)]
        return self._pack_fp8_i32x8(f32)

    def _pv_fp8_direct(self, p_fp8, v_v, v_o):
        v_o = _anchor_v_o(self.traits, v_o)
        for dc in range_constexpr(self.traits.D_CHUNKS):
            v_op = self._v_concat_i32x8(v_v, dc)
            v_o[dc] = self._mfma_acc_fp8_wide(v_op, p_fp8, v_o[dc])
        return v_o

    def pv(self, v_p, v_v, v_o):
        return self._pv_fp8_direct(v_p, v_v, v_o)


class DualwaveFp8KvGmemToLdsLoader(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def load_k(self, tile_start, buf_id):
        """DMA one K tile into LDS, one pass per head-dim band.

        A band's LDS line is this wave's n-rows of `chunk` bytes, row-contiguous,
        so the QK read indexes it as (band, row, 64-byte slice). The 64-byte tail
        band at head_dim 192 runs on the low 32 lanes and moves no padding.
        """
        traits = self.traits
        eb = traits.ELEM_BYTES
        k_lds_byte_base = self.lds_kv_base_idx + self.k_buf_base(buf_id) * eb
        rows_per_wave = -(-traits.BLOCK_N // traits.NUM_WAVES)
        for d in range_constexpr(self.NUM_DMA_K):
            lanes_per_row = traits.K_BAND_CHUNK[d] // traits.VEC_KV
            slots = rows_per_wave * lanes_per_row
            band_base = (
                k_lds_byte_base
                + traits.K_BAND_BASE[d] * eb
                + self.wave_id_uni * (traits.K_BAND_LINE_STRIDE[d] * eb)
            )
            for pas in range_constexpr(-(-slots // traits.WARP_SIZE)):
                slot = self.lane_in_warp + (pas * traits.WARP_SIZE)
                n_in_tile = (slot // lanes_per_row) * traits.NUM_WAVES + self.wave_id
                global_d = (
                    slot % lanes_per_row
                ) * traits.VEC_KV + traits.K_BAND_GLOBAL_D[d]
                src_elem = (
                    self.kv_gmem_elem_offset + n_in_tile * self.stride_kv_n_v + global_d
                )
                lds_addr = band_base + fx.Index(
                    pas * traits.WARP_SIZE * traits.VEC_KV * eb
                )
                active = min(slots - pas * traits.WARP_SIZE, traits.WARP_SIZE)
                if const_expr(active == traits.WARP_SIZE):
                    self.buffer_load_lds_128(
                        self.k_div, lds_addr, src_elem, tile_start * self.stride_kv_n_v
                    )
                else:
                    self._load_k_band_partial_wave(
                        lds_addr, src_elem, tile_start, active
                    )

    def _load_k_band_partial_wave(self, lds_addr, src_elem, tile_start, active_lanes):
        soffset = tile_start * self.stride_kv_n_v
        k_div = self.k_div

        @flyc.jit
        def _run():
            if self.lane_in_warp < active_lanes:
                self.buffer_load_lds_128(k_div, lds_addr, src_elem, soffset)

        _run()

    def load_v(self, tile_start, buf_id):
        self._stage_v_fp8_block_dma(tile_start, buf_id)

    def _stage_v_fp8_block_dma(self, tile_start, buf_id):
        traits = self.traits
        nbands = traits.HEAD_DIM_V // 16
        v_tile_bytes = (traits.BLOCK_N // 8) * nbands * 128
        buf_off = buf_id * v_tile_bytes
        aligned_base = ((self.lds_vt_base_idx + 127) // 128) * 128
        # The tile is BLOCK_N * nbands 16-byte slots, and one DMA instruction moves a
        # whole wave of them. Hand out instructions, not row-groups: a wave's LDS
        # destination is then always a full WARP_SIZE*16 span, so nothing has to be
        # masked off inside a wave. buffer_load...lds strides the LDS write by lane
        # regardless of exec, so an intra-wave mask would still write past the span.
        per_dma = traits.WARP_SIZE * traits.VEC_KV * traits.ELEM_BYTES
        slots_per_group = 8 * nbands
        num_dma = (traits.BLOCK_N * nbands * 16) // per_dma
        passes = -(-num_dma // traits.NUM_WAVES)
        for pas in range_constexpr(passes):
            dma_id = self.wave_id_uni + (pas * traits.NUM_WAVES)
            slot = dma_id * traits.WARP_SIZE + self.lane
            lds_addr = aligned_base + fx.Index(buf_off) + dma_id * per_dma
            grp = slot // slots_per_group
            rem = slot % slots_per_group
            dest_n = fx.Int32(grp * 8 + rem % 8)
            w16 = dest_n % fx.Int32(16)
            c_add = (w16 >= fx.Int32(4)) & (w16 < fx.Int32(8))
            c_sub = (w16 >= fx.Int32(8)) & (w16 < fx.Int32(12))
            n = (
                dest_n
                + c_add.select(fx.Int32(4), fx.Int32(0))
                - c_sub.select(fx.Int32(4), fx.Int32(0))
            )
            d_block = rem // 8
            src_elem = (
                self.v_gmem_elem_offset + fx.Index(n) * self.stride_v_n_v + d_block * 16
            )
            if const_expr(num_dma % traits.NUM_WAVES == 0 or pas < passes - 1):
                self.buffer_load_lds_128(
                    self.v_div, lds_addr, src_elem, tile_start * self.stride_v_n_v
                )
            else:
                self._load_v_group_if_in_tile(
                    lds_addr, src_elem, tile_start, dma_id, num_dma
                )

    def _load_v_group_if_in_tile(self, lds_addr, src_elem, tile_start, grp, groups):
        soffset = tile_start * self.stride_v_n_v
        v_div = self.v_div

        @flyc.jit
        def _run():
            if grp < groups:
                self.buffer_load_lds_128(v_div, lds_addr, src_elem, soffset)

        _run()


class DualwaveFp8KvLdsToVgprLoader(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def load_k(self, buf_id):
        # Read K in the wide 32x32x64 QK operand layout (32 contiguous head-dim/lane,
        # two N-strips, two head-dim halves).
        traits = self.traits
        k_base = self.k_buf_base(buf_id)
        d_base = self.lane_div_32 * 32
        n_lo = self.lane_mod_32
        n_hi = self.lane_mod_32 + 32

        rows_per_line = traits.NUM_WAVES

        def _read_strip(key):
            out = []
            for ws in range_constexpr(traits.HEAD_DIM // 64):
                b = traits.K_WS_BAND[ws]
                line = (key % rows_per_line) * traits.K_BAND_LINE_STRIDE[b]
                row = line + (key // rows_per_line) * traits.K_BAND_CHUNK[b]
                addr = (
                    k_base + traits.K_BAND_BASE[b] + row + traits.K_WS_OFF[ws] + d_base
                )
                out.append(self.read_i32x8_lds(self.lds_kv_base_ptr, addr))
            return out

        return (_read_strip(n_lo), _read_strip(n_hi))

    def load_v(self, buf_id):
        return self._load_v_fp8_block(buf_id)

    def _load_v_fp8_block(self, buf_id):
        traits = self.traits
        v_tile_bytes = (traits.BLOCK_N // 8) * (traits.HEAD_DIM_V // 16) * 128
        buf_off = buf_id * v_tile_bytes
        nbands = traits.HEAD_DIM_V // 16
        rh = (self.lane % 32) // 16
        l16 = self.lane % 16
        lane_hi = self.lane // 32
        aligned_base = ((self.lds_vt_base_idx + 127) // 128) * 128
        base = fx.Int32(
            aligned_base + buf_off + rh * 128 + l16 * 8 + lane_hi * (nbands * 128)
        )

        def _tr8(imm):
            r = _ds_read_tr8_b64_imm(self.v2i32_type, base, imm)
            return llvm.bitcast(T.i64, as_mlir_value(Vec(r)))

        packs = [[None] * traits.D_CHUNKS for _ in range(4)]
        for dc in range_constexpr(traits.D_CHUNKS):
            for ks in range_constexpr(4):
                imm0 = (2 * ks * nbands + dc * 2) * 128
                packs[ks][dc] = _tr8(imm0)
        return packs


class DualwaveFp8SoftmaxHelper(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _attn_mask_vec2_imm(
        self, rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32
    ):
        return _attn_mask_vec2_imm(
            rel_i32, neg_inf_i32, thr_x, thr_y, x_ref_i32, y_ref_i32
        )

    def v_s_vec_to_lists(self, v_s):
        return _score_pair_to_lists(v_s)

    def _causal_mask_inplace(self, v_s, tile_idx):
        traits = self.traits
        s_lo, s_hi = v_s
        kv_tile_start = tile_idx * traits.BLOCK_N
        kv_start_i32 = fx.Int32(kv_tile_start)
        lane_off_i32 = fx.Int32(self.lane_div_32) * fx.Int32(4)
        # q_row_i32 is set by init_q_row (called after helper construction), so read
        # it from the live ctx.
        rel_lo_i32 = fx.Int32(
            self.ctx_ref.q_row_i32 + self.delta_i32 - kv_start_i32 - lane_off_i32
        )
        rel_hi_i32 = fx.Int32(rel_lo_i32 - fx.Int32(32))
        neg_inf_i32 = fx.Int32(traits.NEG_INF_F32_BITS)
        pair_thresholds = _causal_pair_thresholds(False)
        _apply_dualwave_causal_mask_pair(s_lo, rel_lo_i32, neg_inf_i32, pair_thresholds)
        _apply_dualwave_causal_mask_pair(s_hi, rel_hi_i32, neg_inf_i32, pair_thresholds)

    def causal_mask_prologue_if_needed(self, v_s, tile_idx=None, kv_end_pos=None):
        if tile_idx is None:
            tile_idx = 0
        if kv_end_pos is None:
            kv_end_pos = self.traits.BLOCK_N

        @flyc.jit
        def _run(v_s, tile_idx=tile_idx, kv_end_pos=kv_end_pos):
            s_lo, s_hi = v_s
            if self.ctx_ref.q_start_pos_i32 + self.delta_i32 < fx.Int32(kv_end_pos):
                lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                self._causal_mask_inplace((lo_list, hi_list), tile_idx)
                s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _run(v_s)

    def causal_mask_pair_if_needed(self, v_s_a, v_s_b, tile_a):
        """Causal-mask a BN128 tile pair under one scalar-uniform branch.

        Branching per sub-tile would put two scf.if regions in the body on every
        iteration, splitting it into five basic blocks and blocking the QK/softmax/PV
        interleave the sched_group_barriers ask for. Branching once on the pair's end
        position keeps every strictly-below-diagonal pair a single straight-line block.
        Masking sub-tile `a` inside the taken branch even when only `b` needs it is a
        no-op beyond the VALU compares, and happens in at most one pair per q-block.

        This replaces seq_pad_mask_if_needed: with delta = seqlen_kv - seqlen_q the
        largest key any row may attend to is seqlen_kv - 1, so every padding column is
        already strictly above the diagonal.
        """
        traits = self.traits
        kv_end_pos = (tile_a + 2) * traits.BLOCK_N

        @flyc.jit
        def _run(v_s_a, v_s_b, tile_a=tile_a, kv_end_pos=kv_end_pos):
            a_lo, a_hi = v_s_a
            b_lo, b_hi = v_s_b
            if self.ctx_ref.q_start_pos_i32 + self.delta_i32 < fx.Int32(kv_end_pos):
                a_l, a_h = self.v_s_vec_to_lists(v_s_a)
                self._causal_mask_inplace((a_l, a_h), tile_a)
                a_lo, a_hi = _score_lists_to_vecs((a_l, a_h))
                b_l, b_h = self.v_s_vec_to_lists(v_s_b)
                self._causal_mask_inplace((b_l, b_h), tile_a + 1)
                b_lo, b_hi = _score_lists_to_vecs((b_l, b_h))
            return a_lo, a_hi, b_lo, b_hi

        a_lo, a_hi, b_lo, b_hi = _run(v_s_a, v_s_b)
        return (a_lo, a_hi), (b_lo, b_hi)

    def _seq_pad_mask_inplace(self, v_s_lists, tile_idx):
        traits = self.traits
        s_lo, s_hi = v_s_lists
        kv_tile_start = tile_idx * traits.BLOCK_N
        col_base = fx.Int32(kv_tile_start) + fx.Int32(self.lane_div_32) * fx.Int32(4)
        for r in range_constexpr(16):
            thr = (r // 4) * 8 + (r % 4)
            col_lo = col_base + fx.Int32(thr)
            col_hi = col_lo + fx.Int32(32)
            s_lo[r] = (col_lo < self.seqlen_kv_i32).select(s_lo[r], self.c_neg_inf)
            s_hi[r] = (col_hi < self.seqlen_kv_i32).select(s_hi[r], self.c_neg_inf)

    def seq_pad_mask_if_needed(self, v_s, tile_idx=None):
        if tile_idx is None:
            tile_idx = 0

        @flyc.jit
        def _run(v_s, tile_idx=tile_idx):
            s_lo, s_hi = v_s
            kv_tile_end = (tile_idx + 1) * self.traits.BLOCK_N
            if fx.Int32(kv_tile_end) > self.seqlen_kv_i32:
                lo_list, hi_list = self.v_s_vec_to_lists(v_s)
                self._seq_pad_mask_inplace((lo_list, hi_list), tile_idx)
                s_lo, s_hi = _score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _run(v_s)

    def reduce_max(self, v_s):
        return _score_pair_max(v_s, self.c_neg_inf, self.fm_fast)

    def max2(self, a, b):
        return fx.maxnumf(a, b)

    def floor_masked_max(self, row_max):
        return fx.maxnumf(row_max, self.c_neg_floor)

    # log2 of e4m3's largest finite value, 448.
    _P_HEADROOM_LOG2 = 8.807354922057604

    def sub_m(self, v_s, row_max):
        # P is cast to e4m3, whose smallest subnormal is 2**-9, so a softmax
        # over thousands of keys loses its tail to flush-to-zero -- while l_row,
        # summed before the cast, still counts it. Scaling P up first uses the
        # format's whole range; l_row scales with it, so the output is unchanged
        # apart from the tail that survives. Free: it rides the FMA's addend.
        #
        # Available headroom is bounded by how large exp2 gets: the lazy path
        # holds the running max until a tile exceeds it by RESCALE_THRESHOLD, so
        # exp2 <= 2**THRESHOLD there; the eager path rebases every tile.
        headroom = self._P_HEADROOM_LOG2
        if const_expr(self.traits.DUALWAVE_SWP_LAZY_RESCALE):
            headroom -= self.traits.DUALWAVE_SWP_RESCALE_THRESHOLD
        bias = fx.Float32(headroom) if headroom > 0.0 else None
        return _scale_sub_score_pair(
            v_s, row_max, self.c_logit_scale, self.c_zero_f, self.fm_fast, bias
        )

    def exp2(self, v_s, start):
        return _exp2_score_slice(v_s, start)

    def tile_sum(self, v_p):
        return _score_pair_sum(v_p, self.c_zero_f, self.fm_fast)

    def reduce_sum(self, l_row, v_p):
        return l_row + self.tile_sum(v_p)

    def cast_p(self, v_p):
        # Pack the finished softmax probabilities into v8 bf16 P packs for PV.
        return _pack_p_v8_slices(self.traits, v_p, self.bf16_trunc_pack_v8)

    def scale_o(self, v_o, scale_scalar):
        _scale_o_accs(v_o, scale_scalar, self.traits)

    def scale_v_p(self, v_p, scale_scalar):
        # P is v8 bf16 (HIPREC): ext to f32, scale, repack bf16.
        p_lo, p_hi = v_p
        out_lo, out_hi = [], []
        for src, dst in ((p_lo, out_lo), (p_hi, out_hi)):
            for pk in src:
                f32 = Vec(
                    llvm.FPExtOp(
                        Vec.make_type(8, fx.Float32), as_mlir_value(pk)
                    ).result,
                    (8,),
                    fx.Float32,
                )
                scaled = [fx.Float32(f32[i]) * scale_scalar for i in range(8)]
                dst.append(self.bf16_trunc_pack_v8(scaled))
        return out_lo, out_hi

    def anchor_v_p(self, v_p):
        return _anchor_v_p(self.traits, v_p, elem_dtype=self.p_elem)

    def anchor_v_o(self, v_o):
        return _anchor_v_o(self.traits, v_o)

    def anchor_scalar_f32(self, x):
        return _anchor_scalar_f32(x)

    def safe_l_inv(self, l_row):
        return _safe_l_inv(l_row, self.c_zero_f)

    def rescale_from_tile_max(self, m_row, m_tile_max):
        row_max = fx.maxnumf(m_row, m_tile_max)
        diff_scaled = (m_row - row_max) * self.c_logit_scale
        rescale = rocdl.exp2(T.f32, as_mlir_value(diff_scaled))
        return row_max, rescale

    def apply_l_rescale(self, l_row, rescale):
        return l_row * rescale

    def rescale_o(self, v_o, m_row, l_row, m_tile_max, v_p):
        m_new, corr = self.rescale_from_tile_max(m_row, m_tile_max)
        self.scale_o(v_o, corr)
        v_o = self.anchor_v_o(v_o)
        v_p = self.scale_v_p(v_p, corr)
        l_row = self.apply_l_rescale(l_row, corr)
        return v_o, m_new, l_row, v_p

    def v_p_to_vec32(self, v_p):
        # P packs are (p_lo[0..1], p_hi[0..1]) v8 bf16; concat into one v32 SSA value
        # for the scf.if loop-carry.
        return _v_p_to_vec32(v_p)

    def v_vec32_to_p(self, v_p_all):
        return _v_vec32_to_p(self.traits, v_p_all, elem_dtype=self.p_elem)

    def _lazy_correction(self, v_o, m_row, m_tile_max):
        """Monotonic: a downward rebase gives corr > 1 and repeated ones overflow."""
        m_new, corr = self.rescale_from_tile_max(m_row, m_tile_max)
        scaled_accs = list(v_o)
        self.scale_o(scaled_accs, corr)
        return (
            m_new,
            corr,
            [as_mlir_value(scaled_accs[i]) for i in range(self.traits.D_CHUNKS)],
        )

    def lazy_rescale_o(self, v_o, m_row, l_row, m_tile_max, v_p):
        @flyc.jit
        def _run(v_o, m_row, l_row, m_tile_max, v_p):
            m_diff = m_tile_max - m_row
            m_diff_scaled = m_diff * self.c_logit_scale
            below = fx.Float32(m_diff_scaled) <= self.c_rescale_thr_f
            ballot = rocdl.ballot(T.i64, as_mlir_value(below))
            all_below = arith.cmpi(
                arith.CmpIPredicate.eq, as_mlir_value(ballot), _read_exec_i64()
            )
            all_below = llvm.intr_expect(
                all_below, arith.constant(1, type=ir.IntegerType.get_signless(1))
            )

            o_out = [
                as_mlir_value(v_o[dc]) for dc in range_constexpr(self.traits.D_CHUNKS)
            ]
            m_out = as_mlir_value(m_row)
            l_out = as_mlir_value(l_row)
            vp_out = self.v_p_to_vec32(v_p)
            if fx.Boolean(all_below):
                pass
            else:
                m_new, corr, scaled_accs = self._lazy_correction(v_o, m_row, m_tile_max)
                o_out = [
                    as_mlir_value(scaled_accs[dc])
                    for dc in range_constexpr(self.traits.D_CHUNKS)
                ]
                vp_out = self.v_p_to_vec32(self.scale_v_p(v_p, corr))
                l_out = as_mlir_value(l_row * corr)
                m_out = self.anchor_scalar_f32(m_new)
            return (o_out, m_out, l_out, self.v_vec32_to_p(vp_out))

        return _run(v_o, m_row, l_row, m_tile_max, v_p)

    def lazy_correct_o(self, v_o, m_row, l_row, m_tile_max):
        @flyc.jit
        def _run(v_o, m_row, l_row, m_tile_max):
            m_diff = m_tile_max - m_row
            m_diff_scaled = m_diff * self.c_logit_scale
            below = fx.Float32(m_diff_scaled) <= self.c_rescale_thr_f
            ballot = rocdl.ballot(T.i64, as_mlir_value(below))
            all_below = arith.cmpi(
                arith.CmpIPredicate.eq, as_mlir_value(ballot), _read_exec_i64()
            )
            all_below = llvm.intr_expect(
                all_below, arith.constant(1, type=ir.IntegerType.get_signless(1))
            )

            o_out = [
                as_mlir_value(v_o[dc]) for dc in range_constexpr(self.traits.D_CHUNKS)
            ]
            m_out = as_mlir_value(m_row)
            l_out = as_mlir_value(l_row)
            if fx.Boolean(all_below):
                pass
            else:
                m_new, corr, scaled_accs = self._lazy_correction(v_o, m_row, m_tile_max)
                o_out = [
                    as_mlir_value(scaled_accs[dc])
                    for dc in range_constexpr(self.traits.D_CHUNKS)
                ]
                l_out = as_mlir_value(l_row * corr)
                m_out = self.anchor_scalar_f32(m_new)
            return (o_out, m_out, l_out)

        return _run(v_o, m_row, l_row, m_tile_max)


class DualwaveFp8StoreHelper(DualwaveFp8KernelContext):
    def __init__(self, ctx):
        super().__init__(ctx)

    def _o_pack_2dw(self, v_o, dc, store_group):
        r_base = store_group * 4
        lo = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base], Vec(v_o[dc])[r_base + 1])
        hi = rocdl.cvt_pk_bf16_f32(Vec(v_o[dc])[r_base + 2], Vec(v_o[dc])[r_base + 3])
        return lo, hi

    def _swap_half_partner(self, dw):
        pair_i32_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
        swapped = rocdl.permlane32_swap(
            pair_i32_ty, as_mlir_value(dw), as_mlir_value(dw), False, False
        )
        lo_res = llvm.extractvalue(T.i32, swapped, [0])
        hi_res = llvm.extractvalue(T.i32, swapped, [1])
        return (self.lane_div_32 != 0).select(lo_res, hi_res)

    def _packed_o_128_dwords(self, v_o, dc, g):
        is_hi_half = self.lane_div_32 != 0
        d0_a, d1_a = self._o_pack_2dw(v_o, dc, 2 * g)
        d0_b, d1_b = self._o_pack_2dw(v_o, dc, 2 * g + 1)
        y0_a, y1_a = self._swap_half_partner(d0_a), self._swap_half_partner(d1_a)
        y0_b, y1_b = self._swap_half_partner(d0_b), self._swap_half_partner(d1_b)
        w0 = is_hi_half.select(y0_b, as_mlir_value(d0_a))
        w1 = is_hi_half.select(y1_b, as_mlir_value(d1_a))
        w2 = is_hi_half.select(as_mlir_value(d0_b), y0_a)
        w3 = is_hi_half.select(as_mlir_value(d1_b), y1_a)
        return w0, w1, w2, w3

    def _packed_o_128_vec(self, v_o, dc, g):
        return Vec.from_elements(
            [fx.Int32(w) for w in self._packed_o_128_dwords(v_o, dc, g)], fx.Int32
        )

    def store_final_o(self, v_o, q_row):
        for dc in range_constexpr(self.traits.D_CHUNKS):
            for g in range_constexpr(2):
                o_pack = self._packed_o_128_vec(v_o, dc, g)
                d_col = (dc * self.traits.D_CHUNK) + (2 * g + self.lane_div_32) * 8
                o_global = self.global_idx_o(q_row, d_col)
                self.buffer_store_128(o_pack, o_global)

    def store_splitk_partial_o(self, v_o, m_row, l_row, q_row):
        m_row = fx.Float32(m_row) * self.c_logit_scale
        split_z = self.batch_idx * self.traits.NUM_KV_SPLITS + self.split_idx
        o_part_row_base = (
            (split_z * self.traits.NUM_HEADS_Q + self.q_head_idx) * self.seq_len_v
            + q_row
        ) * (self.traits.HEAD_DIM_V // 2)
        grid_z = fx.Index(gpu.grid_dim.z)
        mrow_base = (
            grid_z
            * self.traits.NUM_HEADS_Q
            * self.seq_len_v
            * (self.traits.HEAD_DIM_V // 2)
        )
        lrow_base = mrow_base + grid_z * self.traits.NUM_HEADS_Q * self.seq_len_v
        ml_row_idx = (
            split_z * self.traits.NUM_HEADS_Q + self.q_head_idx
        ) * self.seq_len_v + q_row

        @flyc.jit
        def _store_splitk_partial_if_qrow():
            if q_row < self.seq_len_v:
                for dc in range_constexpr(self.traits.D_CHUNKS):
                    for g in range_constexpr(2):
                        dw_col = (
                            dc * (self.traits.D_CHUNK // 2)
                            + (2 * g + self.lane_div_32) * 4
                        )
                        self.ws_store_quad_i32(
                            self._packed_o_128_dwords(v_o, dc, g),
                            o_part_row_base + dw_col,
                        )
                if self.lane < 32:
                    self.ws_store_f32(m_row, mrow_base + ml_row_idx)
                    self.ws_store_f32(l_row, lrow_base + ml_row_idx)

        _store_splitk_partial_if_qrow()

    def store_empty_split(self):
        @flyc.jit
        def _store_empty_split():
            if self.max_num_tiles < self.split_t0 + 4:
                q_row_e = self.q_start + self.wave_q_offset + self.lane_mod_32
                split_z_e = self.batch_idx * self.traits.NUM_KV_SPLITS + self.split_idx
                o_row_base_e = (
                    (split_z_e * self.traits.NUM_HEADS_Q + self.q_head_idx)
                    * self.seq_len_v
                    + q_row_e
                ) * (self.traits.HEAD_DIM_V // 2)
                grid_z_e = fx.Index(gpu.grid_dim.z)
                mrow_base_e = (
                    grid_z_e
                    * self.traits.NUM_HEADS_Q
                    * self.seq_len_v
                    * (self.traits.HEAD_DIM_V // 2)
                )
                lrow_base_e = (
                    mrow_base_e + grid_z_e * self.traits.NUM_HEADS_Q * self.seq_len_v
                )
                ml_row_e = (
                    split_z_e * self.traits.NUM_HEADS_Q + self.q_head_idx
                ) * self.seq_len_v + q_row_e
                if q_row_e < self.seq_len_v:
                    c_zero_i = fx.Int32(0)
                    for dc in range_constexpr(self.traits.D_CHUNKS):
                        for g in range_constexpr(2):
                            dw_col = (
                                dc * (self.traits.D_CHUNK // 2)
                                + (2 * g + self.lane_div_32) * 4
                            )
                            self.ws_store_quad_i32(
                                [c_zero_i, c_zero_i, c_zero_i, c_zero_i],
                                o_row_base_e + dw_col,
                            )
                    if self.lane < 32:
                        self.ws_store_f32(fx.Float32(-1e30), mrow_base_e + ml_row_e)
                        self.ws_store_f32(self.c_zero_f, lrow_base_e + ml_row_e)

        _store_empty_split()
