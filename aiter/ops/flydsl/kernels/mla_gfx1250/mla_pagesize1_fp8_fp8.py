# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""gfx1250 page-size-1 FP8 persistent MLA decode stage-1 kernel."""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator
from .. import buffer_ops
from ..gemm_common_gfx1250 import make_lds_copy_ops
from .mla_common import (
    _dwordx4_iter,
    _instruction_prefetch,
    _xor16_f32,
    make_global_load_b128,
)

BLOCK_THREADS = 256
WAVE_SIZE = 32
NUM_WAVES = BLOCK_THREADS // WAVE_SIZE
HEADS_PER_WAVE = 16
NUM_Q_HEADS = NUM_WAVES * HEADS_PER_WAVE
QK_NOPE_HEAD_DIM = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM = QK_NOPE_HEAD_DIM
Q_HEAD_STRIDE = QK_HEAD_DIM
Q_ROW_STRIDE = NUM_Q_HEADS * Q_HEAD_STRIDE
Q_NOPE_FRAGMENT_COUNT = QK_NOPE_HEAD_DIM // 128
Q_NOPE_FRAGMENT_DWORDS = 16
Q_ROPE_FRAGMENT_DWORDS = 8
QK_ACC_DWORDS = 8
QK_TILE_DS_OPS = Q_NOPE_FRAGMENT_COUNT * 4 + 2
QK_WMMA_PER_N_TILE = Q_NOPE_FRAGMENT_COUNT + 1
QK_DS_READ_SCHEDULE = (4, 4, 4, 3, 3)
PV_D_TILES = V_HEAD_DIM // 16
PV_ACC_DWORDS = 8
PV_REG_D_TILES = PV_D_TILES
PV_LOAD_DEPTH = 8
PAGE_SIZE = 1
KV_TILE_TOKENS = 64
KV_GATHER_ROWS_PER_WAVE = KV_TILE_TOKENS // NUM_WAVES
KV_N_TILES = KV_TILE_TOKENS // 16
PACKED_PROB_WORDS = KV_TILE_TOKENS // 2 // 4
KV_NOPE_ROW_STRIDE = QK_NOPE_HEAD_DIM + 16
KV_NOPE_SLOT_BYTES = KV_TILE_TOKENS * KV_NOPE_ROW_STRIDE
KV_ROPE_SLOT_OFFSET = KV_NOPE_SLOT_BYTES
KV_ROPE_SLOT_BYTES = KV_TILE_TOKENS * QK_ROPE_HEAD_DIM
KV_SLOT_BYTES = KV_NOPE_SLOT_BYTES + KV_ROPE_SLOT_BYTES
KV_RING_STAGES = 5
KV_RING_BYTES = KV_RING_STAGES * KV_SLOT_BYTES
OUTPUT_LDS_WAVE_BYTES = HEADS_PER_WAVE * V_HEAD_DIM * 4
OUTPUT_LDS_BYTES = NUM_WAVES * OUTPUT_LDS_WAVE_BYTES
LDS_TOTAL_BYTES = max(KV_RING_BYTES, OUTPUT_LDS_BYTES)
INSTRUCTION_PREFETCH_PAGES = 4
LOG2E = math.log2(math.e)


@flyc.jit
def launch_mla_pagesize1_fp8_fp8(
    ptr_r: fx.Pointer,
    ptr_lse: fx.Pointer,
    ptr_final: fx.Pointer,
    ptr_q: fx.Pointer,
    ptr_kv: fx.Pointer,
    kv_page_indices: fx.Pointer,
    work_indptr: fx.Pointer,
    work_info_set: fx.Pointer,
    softmax_scale: fx.Float32,
    num_pages: fx.Int32,
    num_cus: fx.Constexpr[int],
    lds_size: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    """Launch one persistent 256-thread workgroup per CU."""
    assert (
        LDS_TOTAL_BYTES <= lds_size
    ), f"Kernel requires {LDS_TOTAL_BYTES} bytes LDS but CU budget is {lds_size}"

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        ptr_r: fx.Pointer,
        ptr_lse: fx.Pointer,
        ptr_q: fx.Pointer,
        ptr_kv: fx.Pointer,
        kv_page_indices: fx.Pointer,
        work_indptr: fx.Pointer,
        work_info_set: fx.Pointer,
        softmax_scale: fx.Float32,
        num_pages: fx.Int32,
    ):
        """Persistent stage-1 MLA kernel for one-token KV pages."""
        fm_no_inf = (
            fx.FastMathFlags.nnan
            | fx.FastMathFlags.nsz
            | fx.FastMathFlags.arcp
            | fx.FastMathFlags.contract
            | fx.FastMathFlags.afn
            | fx.FastMathFlags.reassoc
        )
        rocdl.disable_xdl_arb_stall()
        _instruction_prefetch(INSTRUCTION_PREFETCH_PAGES)

        smem = SmemAllocator(None, arch="gfx1250", global_sym_name="mla_pagesize1_smem")
        smem.ptr = LDS_TOTAL_BYTES
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            smem.finalize()

        lds_memref = smem.get_base()
        lds_base_idx = fx.Index(
            memref_dialect.extract_aligned_pointer_as_index(lds_memref)
        )
        lds_base = fx.inttoptr(
            fx.PointerType.get(
                elem_ty=fx.Int8.ir_type,
                address_space=fx.AddressSpace.Shared,
                alignment=8,
            ),
            fx.index_cast(T.i32, lds_base_idx),
        )
        lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)
        global_load_b128 = make_global_load_b128()

        kv_pages = fx.Tensor(fx.make_view(ptr_kv, fx.make_layout(QK_HEAD_DIM, 1)))
        page_indices_addr = fx.Int64(fx.ptrtoint(kv_page_indices))
        page_indices_nbytes = fx.Int64(num_pages) * 4
        page_indices_rsrc = buffer_ops.create_buffer_resource_from_addr(
            page_indices_addr.ir_value(),
            num_records_bytes=page_indices_nbytes.ir_value(),
        )

        tid = fx.Int32(fx.thread_idx.x)
        worker_idx = fx.Int32(fx.block_idx.x)
        wave_id = rocdl.readfirstlane(T.i32, tid >> 5)
        lane_id = tid & (WAVE_SIZE - 1)
        head_in_wave = lane_id & (HEADS_PER_WAVE - 1)
        lane_half = lane_id >> 4
        head = wave_id * HEADS_PER_WAVE + head_in_wave

        def _concat_wmma_operand(chunks):
            v01 = chunks[0].shuffle(chunks[1], list(range(8)))
            v23 = chunks[2].shuffle(chunks[3], list(range(8)))
            return v01.shuffle(v23, list(range(16)))

        def _concat_wmma_operand_k64(chunks):
            return chunks[0].shuffle(chunks[1], list(range(8)))

        def _rmem_i32(n, value):
            fragment = fx.make_rmem_tensor(n, fx.Int32)
            fragment.store(value)
            return fragment

        qk_wmma_k128 = fx.make_mma_atom(
            fx.rocdl.WMMA(
                16,
                16,
                128,
                fx.Float8E4M3FN,
                fx.Float32,
            )
        )
        qk_wmma_k64 = fx.make_mma_atom(
            fx.rocdl.WMMA(
                16,
                16,
                64,
                fx.Float8E4M3FN,
                fx.Float32,
            )
        )

        zero_indices = [fx.Int32(0) for _ in range_constexpr(KV_GATHER_ROWS_PER_WAVE)]
        nope_descriptor_template = tdm_ops.make_tensor_gather_descriptor(
            global_ptr=kv_pages,
            lds_memref=lds_memref,
            row_indices=zero_indices,
            row_width=QK_NOPE_HEAD_DIM,
            tensor_dim0=QK_HEAD_DIM,
            tensor_dim1=num_pages,
            stride=QK_HEAD_DIM,
            elem_bytes=1,
            pad_interval=QK_NOPE_HEAD_DIM,
            pad_amount=KV_NOPE_ROW_STRIDE - QK_NOPE_HEAD_DIM,
            index_size=32,
            gather_tile_dim1=KV_GATHER_ROWS_PER_WAVE,
            lds_byte_offset=fx.Index(0),
        )
        rope_descriptor_template = tdm_ops.make_tensor_gather_descriptor(
            global_ptr=kv_pages,
            lds_memref=lds_memref,
            row_indices=zero_indices,
            row_width=QK_ROPE_HEAD_DIM,
            tensor_dim0=QK_HEAD_DIM,
            tensor_dim1=num_pages,
            stride=QK_HEAD_DIM,
            elem_bytes=1,
            index_size=32,
            gather_tile_dim1=KV_GATHER_ROWS_PER_WAVE,
            lds_byte_offset=fx.Index(0),
            global_byte_offset=fx.Int64(QK_NOPE_HEAD_DIM),
        )

        def prepare_kv_tile(tile_start, kv_end, raw_slot):
            slot_byte_offset = raw_slot * KV_SLOT_BYTES
            wave_token_start = tile_start + wave_id * KV_GATHER_ROWS_PER_WAVE
            page_indices_lo = Vec(
                buffer_ops.buffer_load(
                    page_indices_rsrc,
                    wave_token_start,
                    vec_width=4,
                    is_scalar=True,
                )
            )
            page_indices_hi = Vec(
                buffer_ops.buffer_load(
                    page_indices_rsrc,
                    wave_token_start + 4,
                    vec_width=4,
                    is_scalar=True,
                )
            )
            row_indices = []
            for i in range_constexpr(KV_GATHER_ROWS_PER_WAVE):
                token_position = wave_token_start + i
                is_valid = token_position < kv_end
                if const_expr(i < 4):
                    physical_page = page_indices_lo[i]
                else:
                    physical_page = page_indices_hi[i - 4]
                row_indices.append(is_valid.select(physical_page, num_pages))

            nope_descriptor = tdm_ops.make_tensor_gather_descriptor(
                global_ptr=kv_pages,
                lds_memref=lds_memref,
                row_indices=row_indices,
                row_width=QK_NOPE_HEAD_DIM,
                tensor_dim0=QK_HEAD_DIM,
                tensor_dim1=num_pages,
                stride=QK_HEAD_DIM,
                elem_bytes=1,
                pad_interval=QK_NOPE_HEAD_DIM,
                pad_amount=KV_NOPE_ROW_STRIDE - QK_NOPE_HEAD_DIM,
                index_size=32,
                gather_tile_dim1=KV_GATHER_ROWS_PER_WAVE,
                lds_byte_offset=fx.Index(
                    slot_byte_offset
                    + wave_id * (KV_GATHER_ROWS_PER_WAVE * KV_NOPE_ROW_STRIDE)
                ),
            )
            rope_descriptor = tdm_ops.make_tensor_gather_descriptor(
                global_ptr=kv_pages,
                lds_memref=lds_memref,
                row_indices=row_indices,
                row_width=QK_ROPE_HEAD_DIM,
                tensor_dim0=QK_HEAD_DIM,
                tensor_dim1=num_pages,
                stride=QK_HEAD_DIM,
                elem_bytes=1,
                index_size=32,
                gather_tile_dim1=KV_GATHER_ROWS_PER_WAVE,
                lds_byte_offset=fx.Index(
                    slot_byte_offset
                    + KV_ROPE_SLOT_OFFSET
                    + wave_id * (KV_GATHER_ROWS_PER_WAVE * QK_ROPE_HEAD_DIM)
                ),
                global_byte_offset=fx.Int64(QK_NOPE_HEAD_DIM),
            )
            nope_descriptor = tdm_ops.TDMGatherDescriptor(
                dgroup0=nope_descriptor.dgroup0,
                dgroup1=nope_descriptor_template.dgroup1,
                dgroup2=nope_descriptor.dgroup2,
                dgroup3=nope_descriptor.dgroup3,
            )
            rope_descriptor = tdm_ops.TDMGatherDescriptor(
                dgroup0=rope_descriptor.dgroup0,
                dgroup1=rope_descriptor_template.dgroup1,
                dgroup2=rope_descriptor.dgroup2,
                dgroup3=rope_descriptor.dgroup3,
            )
            return nope_descriptor, rope_descriptor

        def issue_prepared_kv_tile(nope_descriptor, rope_descriptor):
            rocdl.sched_barrier(0)
            tdm_ops.tensor_load_gather(nope_descriptor)
            tdm_ops.tensor_load_gather(rope_descriptor)
            rocdl.sched_barrier(0)

        @flyc.jit
        def issue_kv_tile(tile_start, kv_end, raw_slot):
            nope_descriptor, rope_descriptor = prepare_kv_tile(
                tile_start, kv_end, raw_slot
            )
            issue_prepared_kv_tile(nope_descriptor, rope_descriptor)

        @flyc.jit
        def wait_kv_tile(has_next, has_second_next):
            if has_second_next:
                tdm_ops.tensor_wait(4)
            else:
                if has_next:
                    tdm_ops.tensor_wait(2)
                else:
                    tdm_ops.tensor_wait(0)

        def load_k_tile(raw_slot, n_tile):
            slot_byte_offset = raw_slot * KV_SLOT_BYTES
            k_nope_row = n_tile * 16 + head_in_wave
            nope_row_offset = (
                slot_byte_offset + k_nope_row * KV_NOPE_ROW_STRIDE + lane_half * 16
            )
            nope_groups = []
            for fragment in range_constexpr(Q_NOPE_FRAGMENT_COUNT):
                chunks = []
                fragment_offset = nope_row_offset + fragment * 128
                for chunk in range_constexpr(4):
                    chunks.append(
                        Vec(
                            lds_load_b128(
                                lds_base_idx,
                                fragment_offset + chunk * 32,
                            )
                        )
                    )
                nope_groups.append(chunks)

            rope_row_offset = (
                slot_byte_offset
                + KV_ROPE_SLOT_OFFSET
                + k_nope_row * QK_ROPE_HEAD_DIM
                + lane_half * 16
            )
            rope_chunks = []
            for chunk in range_constexpr(2):
                rope_chunks.append(
                    Vec(
                        lds_load_b128(
                            lds_base_idx,
                            rope_row_offset + chunk * 32,
                        )
                    )
                )
            return nope_groups, rope_chunks

        def load_v_operand(raw_slot, d_tile):
            slot_byte_offset = raw_slot * KV_SLOT_BYTES
            v_row = (lane_id >> 3) * 4 + (lane_id & 3)
            v_col = ((lane_id & 7) >> 2) * 8
            chunks = []
            for token_tile in range_constexpr(KV_N_TILES):
                v_byte_offset = (
                    slot_byte_offset
                    + token_tile * 16 * KV_NOPE_ROW_STRIDE
                    + v_row * KV_NOPE_ROW_STRIDE
                    + v_col
                    + d_tile * 16
                )
                v_ptr = fx.add_offset(lds_base, v_byte_offset)
                chunks.append(
                    Vec(
                        rocdl.ds_load_tr8_b64(
                            T.vec(2, T.i32),
                            v_ptr.llvm_ptr,
                        )
                    )
                )
            v01 = chunks[0].shuffle(chunks[1], list(range(4)))
            v23 = chunks[2].shuffle(chunks[3], list(range(4)))
            return v01.shuffle(v23, list(range(PV_ACC_DWORDS)))

        def accumulate_pending_pv(
            raw_slot,
            probability_words,
            running_outs,
        ):
            rocdl.sched_barrier(0)
            p_operand = _rmem_i32(PACKED_PROB_WORDS, probability_words)

            updated_outs = []
            staged_v = {}
            for d_tile in range_constexpr(PV_LOAD_DEPTH):
                staged_v[d_tile] = load_v_operand(raw_slot, d_tile)
            for d_tile in range_constexpr(PV_D_TILES):
                prefetch_tile = d_tile + PV_LOAD_DEPTH
                if const_expr(prefetch_tile < PV_D_TILES):
                    staged_v[prefetch_tile] = load_v_operand(raw_slot, prefetch_tile)
                    rocdl.s_wait_dscnt(PV_LOAD_DEPTH * KV_N_TILES)
                else:
                    rocdl.s_wait_dscnt((PV_D_TILES - 1 - d_tile) * KV_N_TILES)

                accumulator = fx.make_rmem_tensor(PV_ACC_DWORDS, fx.Float32)
                accumulator.store(running_outs[d_tile])
                v_operand = _rmem_i32(PV_ACC_DWORDS, staged_v.pop(d_tile))
                fx.gemm(
                    qk_wmma_k64,
                    accumulator,
                    v_operand,
                    p_operand,
                    accumulator,
                )
                updated_outs.append(Vec(accumulator.load()))

            rocdl.sched_dsrd(PV_LOAD_DEPTH * KV_N_TILES)
            for d_tile in range_constexpr(PV_D_TILES):
                rocdl.sched_mfma(1)
                if const_expr(d_tile + PV_LOAD_DEPTH < PV_D_TILES):
                    rocdl.sched_dsrd(KV_N_TILES)
            rocdl.sched_barrier(0)
            return updated_outs

        def process_tile(
            tile_start,
            kv_start,
            kv_end,
            q_nope_operands,
            q_rope_operand,
            state,
            mask_tail=True,
        ):
            running_max = fx.Float32(state[0])
            running_sum = fx.Float32(state[1])
            running_outs = [
                Vec(state[2 + d_tile]) for d_tile in range_constexpr(PV_REG_D_TILES)
            ]
            pending_probability_words = Vec(state[2 + PV_REG_D_TILES])
            tile_ordinal = (tile_start - kv_start) // KV_TILE_TOKENS
            raw_slot = tile_ordinal % KV_RING_STAGES
            producer_slot = (raw_slot + 3) % KV_RING_STAGES
            pending_raw_slot = (raw_slot + 4) % KV_RING_STAGES

            next_tile_start = tile_start + KV_TILE_TOKENS
            second_next_tile_start = tile_start + 2 * KV_TILE_TOKENS
            wait_kv_tile(next_tile_start < kv_end, second_next_tile_start < kv_end)
            producer_tile_start = tile_start + 3 * KV_TILE_TOKENS
            has_producer = producer_tile_start < kv_end
            safe_producer_start = has_producer.select(producer_tile_start, tile_start)
            rocdl.s_barrier_signal(-1)
            producer_nope_descriptor, producer_rope_descriptor = prepare_kv_tile(
                safe_producer_start,
                kv_end,
                producer_slot,
            )
            pv_ready_outs = running_outs
            if tile_start > kv_start:
                pv_ready_outs = accumulate_pending_pv(
                    pending_raw_slot,
                    pending_probability_words,
                    running_outs,
                )
            rocdl.s_barrier_wait(-1)

            if has_producer:
                issue_prepared_kv_tile(
                    producer_nope_descriptor,
                    producer_rope_descriptor,
                )

            rocdl.sched_barrier(0)
            qk_accs = []
            pending_k_tile = load_k_tile(raw_slot, 0)
            for n_tile in range_constexpr(KV_N_TILES):
                nope_groups, rope_chunks = pending_k_tile
                if const_expr(n_tile + 1 < KV_N_TILES):
                    pending_k_tile = load_k_tile(raw_slot, n_tile + 1)
                    inflight_next = QK_TILE_DS_OPS
                else:
                    pending_k_tile = None
                    inflight_next = 0

                accumulator = fx.make_rmem_tensor(QK_ACC_DWORDS, fx.Float32)
                accumulator.store(Vec.filled(QK_ACC_DWORDS, 0.0, fx.Float32))
                for fragment in range_constexpr(Q_NOPE_FRAGMENT_COUNT):
                    remaining_ds = (
                        (Q_NOPE_FRAGMENT_COUNT - 1 - fragment) * 4 + 2 + inflight_next
                    )
                    rocdl.s_wait_dscnt(remaining_ds)
                    k_operand = _rmem_i32(
                        Q_NOPE_FRAGMENT_DWORDS,
                        _concat_wmma_operand(nope_groups[fragment]),
                    )
                    fx.gemm(
                        qk_wmma_k128,
                        accumulator,
                        k_operand,
                        q_nope_operands[fragment],
                        accumulator,
                    )
                rocdl.s_wait_dscnt(inflight_next)
                k_rope_operand = _rmem_i32(
                    Q_ROPE_FRAGMENT_DWORDS,
                    _concat_wmma_operand_k64(rope_chunks),
                )
                fx.gemm(
                    qk_wmma_k64,
                    accumulator,
                    k_rope_operand,
                    q_rope_operand,
                    accumulator,
                )
                qk_accs.append(Vec(accumulator.load()))

            rocdl.sched_dsrd(QK_TILE_DS_OPS)
            for n_tile in range_constexpr(KV_N_TILES):
                for mma_slot in range_constexpr(QK_WMMA_PER_N_TILE):
                    rocdl.sched_mfma(1)
                    if const_expr(n_tile + 1 < KV_N_TILES):
                        rocdl.sched_dsrd(QK_DS_READ_SCHEDULE[mma_slot])
            rocdl.sched_barrier(0)

            valid_count = fx.Int32(KV_TILE_TOKENS)
            if const_expr(mask_tail):
                valid_count = kv_end - tile_start
                valid_count = (valid_count > fx.Int32(KV_TILE_TOKENS)).select(
                    fx.Int32(KV_TILE_TOKENS), valid_count
                )
            negative_inf = fx.Float32(float("-inf"))
            masked_scores = []
            for n_tile in range_constexpr(KV_N_TILES):
                tile_scores = []
                for i in range_constexpr(QK_ACC_DWORDS):
                    local_token = n_tile * 16 + lane_half * QK_ACC_DWORDS + i
                    score = qk_accs[n_tile][i] * softmax_scale
                    if const_expr(mask_tail):
                        valid_token = local_token < valid_count
                        tile_scores.append(valid_token.select(score, negative_inf))
                    else:
                        tile_scores.append(score)
                masked_scores.append(tile_scores)

            with fx.fastmath(fm_no_inf):
                local_max = masked_scores[0][0]
                for n_tile in range_constexpr(KV_N_TILES):
                    for i in range_constexpr(QK_ACC_DWORDS):
                        local_max = local_max.maximumf(masked_scores[n_tile][i])
                tile_max = local_max.maximumf(_xor16_f32(local_max))
                new_max = running_max.maximumf(tile_max)
                alpha_arg = (running_max - new_max) * fx.Float32(LOG2E)
            alpha = fx.Float32(rocdl.exp2(T.f32, alpha_arg.ir_value()))

            with fx.fastmath(fm_no_inf):
                neg_new_max = fx.Float32(0.0) - new_max

            probabilities = []
            sum_vector = Vec.filled(QK_ACC_DWORDS, 0.0, fx.Float32)
            for n_tile in range_constexpr(KV_N_TILES):
                with fx.fastmath(fm_no_inf):
                    if const_expr(mask_tail):
                        probability_args = [
                            (masked_scores[n_tile][i] - new_max) * fx.Float32(LOG2E)
                            for i in range_constexpr(QK_ACC_DWORDS)
                        ]
                    else:
                        arg_vector = (
                            qk_accs[n_tile] * softmax_scale + neg_new_max
                        ) * fx.Float32(LOG2E)
                        probability_args = [
                            arg_vector[i] for i in range_constexpr(QK_ACC_DWORDS)
                        ]
                tile_probabilities = []
                for i in range_constexpr(QK_ACC_DWORDS):
                    probability = fx.Float32(
                        rocdl.exp2(T.f32, probability_args[i].ir_value())
                    )
                    probabilities.append(probability)
                    tile_probabilities.append(probability)
                with fx.fastmath(fm_no_inf):
                    tile_vector = Vec.from_elements(
                        tile_probabilities,
                        fx.Float32,
                    )
                    sum_vector = (
                        tile_vector
                        if const_expr(n_tile == 0)
                        else sum_vector + tile_vector
                    )
            local_sum = fx.Float32(0.0)
            with fx.fastmath(fm_no_inf):
                for i in range_constexpr(QK_ACC_DWORDS):
                    local_sum = local_sum + sum_vector[i]
                tile_sum = local_sum + _xor16_f32(local_sum)
                new_sum = running_sum * alpha + tile_sum

            packed_words = []
            for word in range_constexpr(PACKED_PROB_WORDS):
                base = word * 4
                packed = rocdl.cvt_pk_fp8_f32(
                    T.i32,
                    probabilities[base],
                    probabilities[base + 1],
                    fx.Int32(0),
                    0,
                )
                packed = rocdl.cvt_pk_fp8_f32(
                    T.i32,
                    probabilities[base + 2],
                    probabilities[base + 3],
                    packed,
                    1,
                )
                packed_words.append(fx.Int32(packed))

            packed_probability_words = Vec.from_elements(packed_words, fx.Int32)

            with fx.fastmath(fm_no_inf):
                scaled_outs = [
                    pv_ready_outs[d_tile] * alpha
                    for d_tile in range_constexpr(PV_REG_D_TILES)
                ]

            return [new_max, new_sum] + scaled_outs + [packed_probability_words]

        work_start = fx.Int32(rocdl.readfirstlane(T.i32, work_indptr[worker_idx]))
        work_end = fx.Int32(
            rocdl.readfirstlane(T.i32, work_indptr[worker_idx + fx.Int32(1)])
        )

        for work_idx in range(
            work_start,
            work_end,
            fx.Int32(1),
        ):
            work_base = fx.Int64(work_idx) * fx.Int64(8)

            def work_info_scalar(field):
                return fx.Int32(
                    rocdl.readfirstlane(
                        T.i32, work_info_set[work_base + fx.Int64(field)]
                    )
                )

            partial_qo_loc = work_info_scalar(1)
            qo_start = work_info_scalar(2)
            kv_start = work_info_scalar(4)
            kv_end = work_info_scalar(5)

            q_tile_base = fx.Int64(qo_start) * Q_ROW_STRIDE + fx.Int64(wave_id) * (
                HEADS_PER_WAVE * Q_HEAD_STRIDE
            )
            q_words = _dwordx4_iter(fx.add_offset(ptr_q, q_tile_base))
            q_lane_dword = head_in_wave * (Q_HEAD_STRIDE // 4) + lane_half * 4
            q_nope_operands = []
            for fragment in range_constexpr(Q_NOPE_FRAGMENT_COUNT):
                chunks = []
                for chunk in range_constexpr(4):
                    chunks.append(
                        Vec(
                            global_load_b128(
                                q_words,
                                q_lane_dword + fragment * 32 + chunk * 8,
                            )
                        )
                    )
                q_nope_operands.append(
                    _rmem_i32(Q_NOPE_FRAGMENT_DWORDS, _concat_wmma_operand(chunks))
                )
            q_rope_chunks = []
            for chunk in range_constexpr(2):
                q_rope_chunks.append(
                    Vec(
                        global_load_b128(
                            q_words,
                            q_lane_dword + QK_NOPE_HEAD_DIM // 4 + chunk * 8,
                        )
                    )
                )
            q_rope_operand = _rmem_i32(
                Q_ROPE_FRAGMENT_DWORDS,
                _concat_wmma_operand_k64(q_rope_chunks),
            )
            rocdl.sched_barrier(0)

            has_tokens = kv_start < kv_end
            zero_out = Vec.filled(PV_ACC_DWORDS, 0.0, fx.Float32)
            init_state = (
                [fx.Float32(float("-inf")), fx.Float32(0.0)]
                + [zero_out for _ in range_constexpr(PV_REG_D_TILES)]
                + [Vec.filled(PACKED_PROB_WORDS, 0, fx.Int32)]
            )
            remaining_tokens = kv_end - kv_start
            tile_count = (remaining_tokens > fx.Int32(0)).select(
                (remaining_tokens + KV_TILE_TOKENS - 1) // KV_TILE_TOKENS,
                fx.Int32(0),
            )
            if has_tokens:
                issue_kv_tile(kv_start, kv_end, fx.Int32(0))
                second_tile_start = kv_start + KV_TILE_TOKENS
                if second_tile_start < kv_end:
                    issue_kv_tile(second_tile_start, kv_end, fx.Int32(1))
                third_tile_start = kv_start + 2 * KV_TILE_TOKENS
                if third_tile_start < kv_end:
                    issue_kv_tile(third_tile_start, kv_end, fx.Int32(2))

            last_tile_start = kv_start + (tile_count - 1) * KV_TILE_TOKENS
            full_tile_end = has_tokens.select(last_tile_start, kv_start)

            loop_results = init_state
            for tile_start, state in range(
                kv_start,
                full_tile_end,
                fx.Int32(KV_TILE_TOKENS),
                init=init_state,
            ):
                updated_state = process_tile(
                    fx.Int32(tile_start),
                    kv_start,
                    kv_end,
                    q_nope_operands,
                    q_rope_operand,
                    state,
                    mask_tail=False,
                )
                loop_results = yield updated_state

            if has_tokens:
                loop_results = process_tile(
                    last_tile_start,
                    kv_start,
                    kv_end,
                    q_nope_operands,
                    q_rope_operand,
                    loop_results,
                    mask_tail=True,
                )

            reg_output_accs = [
                Vec(loop_results[2 + d_tile])
                for d_tile in range_constexpr(PV_REG_D_TILES)
            ]
            if has_tokens:
                last_raw_slot = (tile_count - 1) % KV_RING_STAGES
                reg_output_accs = accumulate_pending_pv(
                    last_raw_slot,
                    Vec(loop_results[2 + PV_REG_D_TILES]),
                    reg_output_accs,
                )

            running_max = fx.Float32(loop_results[0])
            running_sum = fx.Float32(loop_results[1])
            has_mass = running_sum > fx.Float32(0.0)
            inv_sum = has_mass.select(
                fx.Float32(rocdl.rcp(T.f32, running_sum.ir_value())),
                fx.Float32(0.0),
            )

            tdm_ops.tensor_wait(0)
            gpu.barrier()
            for d_tile in range_constexpr(PV_D_TILES):
                output_values = [
                    reg_output_accs[d_tile][i] * inv_sum
                    for i in range_constexpr(PV_ACC_DWORDS)
                ]
                for half in range_constexpr(2):
                    output_dwords = Vec.from_elements(
                        [output_values[half * 4 + i] for i in range_constexpr(4)],
                        fx.Float32,
                    ).bitcast(fx.Int32)
                    lds_output_offset = (
                        wave_id * OUTPUT_LDS_WAVE_BYTES
                        + head_in_wave * (V_HEAD_DIM * 4)
                        + (d_tile * 16 + lane_half * PV_ACC_DWORDS + half * 4) * 4
                    )
                    lds_store_b128(
                        lds_base_idx,
                        lds_output_offset,
                        output_dwords,
                    )
            rocdl.s_wait_dscnt(0)
            gpu.barrier()

            output_lds_ptr_type = fx.PointerType.get(
                elem_ty=fx.Float32.ir_type,
                address_space=fx.AddressSpace.Shared,
                alignment=16,
            )
            output_lds_ptr = fx.inttoptr(
                output_lds_ptr_type,
                fx.ptrtoint(lds_base),
            )
            output_tile_base = fx.Int64(partial_qo_loc) * (NUM_Q_HEADS * V_HEAD_DIM)
            output_global = fx.Tensor(
                fx.make_view(
                    fx.add_offset(ptr_r, output_tile_base),
                    fx.make_layout(
                        (NUM_Q_HEADS, V_HEAD_DIM),
                        (V_HEAD_DIM, 1),
                    ),
                )
            )
            output_lds = fx.Tensor(
                fx.make_view(
                    output_lds_ptr,
                    fx.make_layout(
                        (NUM_Q_HEADS, V_HEAD_DIM),
                        (V_HEAD_DIM, 1),
                    ),
                )
            )
            output_tdm_atom = fx.rocdl.make_tdm_atom(
                output_global,
                [NUM_Q_HEADS, None],
                strides=[V_HEAD_DIM, None],
                num_warps=NUM_WAVES,
            )
            fx.copy(
                output_tdm_atom,
                output_lds,
                output_global,
            )
            if lane_half == fx.Int32(0):
                lse = has_mass.select(
                    running_max + fmath.log(running_sum),
                    fx.Float32(float("-inf")),
                )
                ptr_lse[partial_qo_loc * NUM_Q_HEADS + head] = lse
            tdm_ops.tensor_wait(0)
            gpu.barrier()

    kernel(
        ptr_r,
        ptr_lse,
        ptr_q,
        ptr_kv,
        kv_page_indices,
        work_indptr,
        work_info_set,
        softmax_scale,
        num_pages,
    ).launch(
        grid=(num_cus, 1, 1),
        block=(BLOCK_THREADS, 1, 1),
        stream=stream,
    )


launch_mla_pagesize1_fp8_fp8.compile_hints = {
    "llvm_options": {
        "amdgpu-expert-scheduling-mode": True,
    },
}
