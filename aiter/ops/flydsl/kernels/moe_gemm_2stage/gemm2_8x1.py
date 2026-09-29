# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""8x1 production implementation: shared entry/primitives, BK128 multiples, K192, K320."""

from functools import cache, partial

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.typing import as_ir_value

from . import common as fxh
from .common import _f32_to_bf16
from .gemm2_common import (
    BufferTensor,
    DownTileOps,
    LdsTensor,
    eltwise_op,
    get_down_device_config,
)

# ==================== Shared entry point and 8x1 implementation ====================


def _build_moe_gemm2_8x1(
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    stage="down",
    alg="splitk",
    E=None,
    USE_ATOMIC_WRITE=True,
    act_quant_type=None,
    tile_k=None,
    activation="silu",
    swiglu_limit=None,
    down_path="default",
    down_output_padding_bytes=None,
    METADATA_TILE_SIZE_M=None,
    _task_table=False,
    _store_cache=2,
):
    # Validate host arguments; K families share only task mapping and launch, not pipeline events.
    del E, activation, swiglu_limit
    assert stage == "down"
    assert alg == "prefill_1x4"
    assert down_path == "8x1"
    assert weight_dtype == "fp8"
    assert weight_quant_type in ("ptpc", "per_tensor")
    if act_quant_type is None:
        act_quant_type = weight_quant_type
    assert (weight_quant_type == "ptpc" and act_quant_type == "ptpc") or (
        weight_quant_type == "per_tensor" and act_quant_type in ("ptpc", "per_tensor")
    ), f"unsupported 8x1 quant combo (weight={weight_quant_type}, act={act_quant_type})"
    assert not USE_ATOMIC_WRITE
    assert BLOCK_TILE_SIZE_M == 256
    assert BLOCK_TILE_SIZE_N == 128
    if METADATA_TILE_SIZE_M is None:
        METADATA_TILE_SIZE_M = BLOCK_TILE_SIZE_M
    assert METADATA_TILE_SIZE_M == BLOCK_TILE_SIZE_M
    assert K in (
        192,
        256,
        320,
        384,
        512,
        640,
    ), "8x1 only supports K=192/256/320/384/512/640"
    if tile_k is None:
        tile_k = 192 if K in (192, 320) else 128
    assert tile_k == 128 or (
        K in (192, 320) and tile_k == 192
    ), "8x1 keeps one K192 block of 192, K320 as 128+192, and BK128 for other K values"
    # K192 uses one block of 192; K320 is fixed at 128+192. Accept callers passing tile_k=128.
    if K in (192, 320):
        tile_k = 192
        assert N > 0
    else:
        assert K % tile_k == 0
    assert N % BLOCK_TILE_SIZE_N == 0
    assert down_output_padding_bytes in (0, 32, 64, 128)
    assert _store_cache in (0, 2), "store aux: 0=normal, 2=SLC"

    BM = 256
    ops = DownTileOps()
    topology, xcc_count = get_down_device_config()
    se_count = xcc_count * 4

    @flyc.kernel(known_block_size=[512, 1, 1])
    def moe_2stage_down_prefill_8x1(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        M: fx.Int32,
    ):
        # Map tasks by XCC/SE/CU; validate before staggering the two groups of four waves.
        tid = fx.Int32(gpu.thread_idx.x)
        lane, wave = tid % 64, tid // 64
        group = fx.Int32(
            rocdl.readfirstlane(
                ir.IntegerType.get_signless(32), as_ir_value(tid // 256)
            )
        )
        group_tid = tid % 256
        valid = fxh.view_as_torch_tensor(p_num_valid_ids, (1,), fx.Int32)[0]
        wg = fx.Int32(gpu.block_idx.y)
        e_idx = wg
        if const_expr(topology):
            tasks = fxh.div_up(fx.Uint32(valid), BM)
            per_se = tasks // se_count
            mapped = per_se * se_count
            u = fx.Uint32(wg)
            xcc, local = u & (xcc_count - 1), u >> 2
            se, within = local & 3, local >> 2
            cu, round_id = within % 5, within // 5
            short, long = per_se // 5, per_se % 5
            rank = cu * short + arith.select(cu < long, cu, long) + round_id
            logical = ((xcc + 2) & (xcc_count - 1)) * (per_se * 4) + se * per_se + rank
            e_idx = fx.Int32(arith.select(u < mapped, logical, u))

        if e_idx * BM < valid:
            # K is a uniform compile-time choice; large emitters remain module-level JIT helpers.
            if const_expr(K == 192):
                _emit_k192_body(
                    p_input,
                    p_weight,
                    p_output,
                    p_sorted_ids,
                    p_sorted_weights,
                    p_sorted_expert_ids,
                    p_w_scale,
                    p_a_scale,
                    M,
                    e_idx,
                    tid,
                    lane,
                    wave,
                    group,
                    N,
                    TOPK,
                    down_output_padding_bytes,
                    weight_quant_type,
                    act_quant_type,
                    _task_table,
                    _store_cache,
                    ops,
                )
            elif const_expr(K == 320):
                _emit_k320_body(
                    p_input,
                    p_weight,
                    p_output,
                    p_sorted_ids,
                    p_sorted_weights,
                    p_sorted_expert_ids,
                    p_w_scale,
                    p_a_scale,
                    M,
                    e_idx,
                    tid,
                    lane,
                    wave,
                    group,
                    group_tid,
                    N,
                    TOPK,
                    down_output_padding_bytes,
                    weight_quant_type,
                    act_quant_type,
                    _task_table,
                    _store_cache,
                    ops,
                )
            else:
                _emit_k128n_body(
                    p_input,
                    p_weight,
                    p_output,
                    p_sorted_ids,
                    p_sorted_weights,
                    p_sorted_expert_ids,
                    p_w_scale,
                    p_a_scale,
                    M,
                    e_idx,
                    tid,
                    lane,
                    wave,
                    group,
                    group_tid,
                    N,
                    K,
                    TOPK,
                    tile_k,
                    down_output_padding_bytes,
                    weight_quant_type,
                    act_quant_type,
                    _task_table,
                    _store_cache,
                    ops,
                )

    @flyc.jit
    def launch_prefill_8x1(
        p_input: fx.Pointer,
        p_weight: fx.Pointer,
        p_output: fx.Pointer,
        p_sorted_ids: fx.Pointer,
        p_sorted_weights: fx.Pointer,
        p_sorted_expert_ids: fx.Pointer,
        p_num_valid_ids: fx.Pointer,
        p_w_scale: fx.Pointer,
        p_a_scale: fx.Pointer,
        M: fx.Int32,
        task_num: fx.Int32,
        stream: fx.Stream,
    ):
        # Preserve the original ABI, launch shape, and compilation options.
        CompilationContext.get_current()
        kernel = moe_2stage_down_prefill_8x1(
            p_input,
            p_weight,
            p_output,
            p_sorted_ids,
            p_sorted_weights,
            p_sorted_expert_ids,
            p_num_valid_ids,
            p_w_scale,
            p_a_scale,
            M,
            value_attrs={"passthrough": [["target-features", "-packed-fp32-ops"]]},
        )
        kernel.launch(grid=(1, task_num, 1), block=(512, 1, 1), stream=stream)

    launch_prefill_8x1.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch_prefill_8x1


# ==================== Shared 8x1 hardware/math primitives ====================

# g=VMEM, r=registers, s=LDS; read issues an asynchronous load, with waits scheduled separately.


def stage_end():
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def priority(value):
    rocdl.sched_barrier(0)
    rocdl.s_setprio(value)
    rocdl.sched_barrier(0)


def cshuffle_plane_offset(row, group, pair):
    # BF16 element offset; move the source-group low bit to a 2KiB plane, keeping 128-bit writes.
    return (
        (row & 1) * 8
        + ((group & 2) ^ (row & 2)) * 8
        + ((pair & 1) ^ ((row >> 2) & 1)) * 32
        + (pair >> 1) * 64
        + (row >> 3) * 128
        + (row & 6) * 128
        + (group & 1) * 1024
    )


def read_b_g2r_packet(weights, byte_offset, scalar_offset, words):
    source = BufferTensor(
        fx.make_view(fx.get_iter(weights), fx.make_layout(words * 4, 1))
    )
    fragment = fx.make_rmem_tensor(fx.make_layout(words, 1), fx.Uint32)

    values = source.load(
        voffset_bytes=fx.Int32(byte_offset), soffset_bytes=fx.Int32(scalar_offset)
    ).bitcast(fx.Uint32)
    fragment.store(values)
    return fragment


def read_a_g2r(a, ids_lds, afragments, copy_atom, lane, wave, k_widths, k_offsets):
    # BM=256: two row groups per wave, spaced by 128; split each 16B load into two K8 pieces.
    packed_inputs = [
        [
            [
                fx.make_rmem_tensor(fx.make_layout(16, 1), fx.Float8E4M3FNUZ)
                for _ in range_constexpr(width // 64)
            ]
            for _ in range_constexpr(2)
        ]
        for width in k_widths
    ]

    for ks in range_constexpr(len(k_widths)):
        for row in range_constexpr(2):
            encoded = ids_lds[wave * 16 + row * 128 + lane % 16].bitcast(fx.Uint32)
            for k64 in range_constexpr(k_widths[ks] // 64):
                offset = k_offsets[ks] + k64 * 64 + (lane // 16) * 16
                # The source depends on the just-loaded route; construct its dynamic subview here.
                source = fxh.atom_tensor(
                    a, (encoded & 0xFFFFFF, encoded >> 24, offset), 128
                )
                packed = packed_inputs[ks][row][k64]
                fx.copy(copy_atom, source, packed)
                values = Vec(packed.load())
                for k8 in range_constexpr(2):
                    part = values.shuffle(values, list(range(k8 * 8, k8 * 8 + 8)))
                    afragments[ks][None, row, (k8, k64)].store(part)


def read_8x1_scale_g2r(n, pair, *, weight_quant_type, scale_buffer, mm, ops):
    if const_expr(weight_quant_type == "ptpc"):
        # fx.copy soffset is in elements; lowering multiplies by 4 for native byte offsets.
        tensor = fx.make_view(
            fx.get_iter(scale_buffer) + pair * 32, fx.make_layout((32, 256), (1, 0))
        )
        # Keep two dwordx4 instructions per pair; copy32 does not merge and breaks VMEM accounting.
        copy_atom = ops.get_buffer_copy_atom(fx.Float32, 128)
        fragment = mm.make_fragment_C(tensor)

        fx.copy(
            copy_atom,
            ops.get_tiled_mma_partition_S(mm, tensor, "C", copy_atom_bits=128),
            ops.get_tiled_mma_retile(mm, fragment, "C", copy_atom=copy_atom),
            soffset=fx.Int32(n * 128),
        )
        return fragment
    else:
        return fx.Float32(1.0)


def pack_8x1_record(pair, scale, *, c, row_scale, weight_quant_type):
    weighted, rows = [], []
    for row in range_constexpr(2):
        for ng in range_constexpr(2 * pair, 2 * pair + 2):
            if const_expr(weight_quant_type == "ptpc"):
                weighted.append(
                    eltwise_op(
                        "v_fma_f32",
                        Vec(c[None, ng, row].load()),
                        Vec(scale[None, ng % 2, row].load()),
                        fx.Float32(0.0),
                    )
                )
            else:
                weighted.append(Vec(c[None, ng, row].load()))
            rows.append(Vec(row_scale[None, ng, row].load()))
    records = [[], []]
    for index in range_constexpr(4):
        packed = _f32_to_bf16(weighted[index] * rows[index]).bitcast(fx.Uint32)
        for element in range_constexpr(packed.numel):
            records[index // 2].append(packed[element])
    return [
        Vec.from_elements(record, fx.Uint32).bitcast(fx.BFloat16) for record in records
    ]


def shuffle_8x1_c_r2s2r(
    n,
    packed,
    row,
    half,
    *,
    scratch_view,
    scratch_base,
    lane,
    lane_group,
    wave,
    out,
    scratch_write,
    scratch_read,
):
    for local_pair in range_constexpr(2):
        pair = half * 2 + local_pair
        offset = scratch_base + cshuffle_plane_offset(lane % 16, lane_group, pair)
        destination = fx.make_view(
            fx.get_iter(scratch_view) + offset, fx.make_layout(8, 1)
        )
        fragment = fx.make_fragment_like(destination)
        fragment.store(packed[pair][row])
        fx.copy(scratch_write, fragment, destination)
    fragments, destinations = [], []
    for oh in range_constexpr(2):
        atom_index = half * 8 + lane % 8
        ng = atom_index // 2
        offset = (
            scratch_base
            + cshuffle_plane_offset(oh * 8 + lane // 8, 2 * (atom_index % 2), ng // 2)
            + (ng % 2) * 4
        )
        pieces = []
        for source_group in range_constexpr(2):
            source = fx.make_view(
                fx.get_iter(scratch_view) + offset + source_group * 1024,
                fx.make_layout(4, 1),
            )
            fragment = fx.make_fragment_like(source)
            fx.copy(scratch_read, source, fragment)
            pieces.append(fragment)
        # Merge the two 8B pieces of one output into read2st64(offset1:4); never pair across oh.
        fx.rocdl.sched_barrier(0)
        fragments.append(pieces)
        out_row = wave * 16 + row * 128 + oh * 8 + lane // 8
        destinations.append(
            (
                n,
                fx.make_view(
                    fx.get_iter(out) + out.layout(atom_index * 8, out_row),
                    fx.make_layout(8, 1),
                ),
            )
        )
    return fragments, destinations


def store_8x1_c_r2g(fragments, destinations, lgkmcnt=0, *, store_atom):
    rocdl.s_waitcnt(lgkmcnt=lgkmcnt)
    for index in range_constexpr(len(fragments)):
        first, second = Vec(fragments[index][0].load()), Vec(fragments[index][1].load())
        result = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
        result.store(first.shuffle(second, list(range(8))))
        output_n, destination = destinations[index]
        # Per-thread output addresses exclude the N loop variable; SGPR soffset supplies it.
        fx.copy(store_atom, result, destination, soffset=fx.Int32(output_n * 128))


def store_8x1_c_tile_r2g(n, packed, *, shuffle_c_r2s2r, store_c_r2g):
    for half in range_constexpr(2):
        for row in range_constexpr(2):
            fragments, destinations = shuffle_c_r2s2r(n, packed, row, half)
            store_c_r2g(fragments, destinations)


def clear_8x1_record(pair, *, c):
    for row in range_constexpr(2):
        for ng in range_constexpr(pair * 2, pair * 2 + 2):
            c[None, ng, row].fill(0)


def mfma_8x1_record(
    weight, ks, pair, *, k_widths, afragments, c, atom, weight_n_group_begin=0
):
    for ki in range_constexpr(k_widths[ks] // 64):
        for ka in range_constexpr(2):
            for row in range_constexpr(2):
                for ng in range_constexpr(2):
                    weight_piece = weight[None, weight_n_group_begin + ng, (ka, ki)]
                    activation = afragments[ks][None, row, (ka, ki)]
                    fx.mma_atom_call(
                        atom,
                        c[None, pair * 2 + ng, row],
                        weight_piece,
                        activation,
                        c[None, pair * 2 + ng, row],
                    )


def schedule_k128_pack(*, weight_quant_type):
    # PTPC has 40 VALU instructions, or 24 after scalar-scale fusion; use only long BK128 packets.
    for index in range_constexpr(16):
        rocdl.sched_group_barrier(0x8, 1, 0)
        if const_expr(weight_quant_type == "ptpc"):
            if const_expr(index < 13):
                rocdl.sched_group_barrier(0x2, 3, 0)
            elif const_expr(index == 13):
                rocdl.sched_group_barrier(0x2, 1, 0)
        else:
            rocdl.sched_group_barrier(0x2, 2 if index < 8 else 1, 0)
    rocdl.sched_barrier(0)


# ==================== Shared 8x1 event accounting and SSA ====================


def packing_events(k, stage, first=False, k_widths=None):
    """Return (packet, previous-N, super-record), not the current MFMA output fragment."""
    assert k != 192, "K192 uses the independent K192 helper"
    assert k in (256, 320, 384, 512, 640), "shared 8x1 schedule does not support this K"
    assert k_widths == (128, 192) if k == 320 else k_widths is None
    ks = (k + 127) // 128
    if k == 320:
        if stage == 0 and not first:
            return ((0, True, 2), (1, True, 3))
        if stage == 2:
            return ((0, False, 0), (1, False, 1))
    else:
        if stage == 0 and not first:
            return ((0, True, 3),)
        if stage == ks - 1:
            return ((1, False, 0),)
        if stage == ks:
            return ((0, False, 1),)
        if stage == 2 * ks - 1:
            return ((1, False, 2),)
    return ()


def output_quarter(k, stage, k_widths=None):
    assert k != 192, "K192 uses the independent K192 helper"
    assert k in (256, 320, 384, 512, 640), "shared 8x1 schedule does not support this K"
    assert k_widths == (128, 192) if k == 320 else k_widths is None
    return stage if stage < 4 else None


@cache
def vmem_wait_schedule(k, n_tiles, ptpc=True, k_widths=None):
    """Count one event per ordinary buffer/global instruction; issue the next B load after waiting.

    The budget is the minimum age of all B/scale values about to be consumed.
    Counting only B is insufficient: some K256 packets use the preceding step's
    scale, tightening 9 to 7. K192 has independent accounting; K320 requires the
    explicit and unique partition (128, 192).
    """
    assert (
        k in (256, 320, 384, 512, 640) and n_tiles >= 1
    ), "shared 8x1 schedule does not support this K or N tile count"
    assert k_widths == (128, 192) if k == 320 else k_widths is None
    widths = k_widths or (128,) * (k // 128)
    ks, events, requests, scales = len(widths), [], {}, {}

    def valid(q):
        # Request q submits Q[q+1]; prune only requests beyond the final consumer.
        return 0 <= q < n_tiles * 2 * ks - 1

    def request(q):
        if valid(q):
            # The final 192 block loads 24B per lane as 16B+8B; constrain submission by the last VMEM.
            target_k = (q % ks + 1) % ks
            events.extend([("B", q)] * (2 if widths[target_k] == 192 else 1))
            requests[q] = len(events) - 1

    request(0)
    request(1)
    result = []
    for q in range(n_tiles * 2 * ks):
        n, stage = divmod(q, 2 * ks)
        scale_count = 2 if ptpc and stage < 4 else 0
        events.extend([("scale", n, stage)] * scale_count)
        if scale_count:
            scales[n, stage] = len(events) - 1
        store_count = (
            2 if n > 0 and output_quarter(k, stage, k_widths) is not None else 0
        )
        events.extend([("store", n, stage)] * store_count)
        required = [requests[q]] if valid(q) else []
        if ptpc:
            for _, previous, record in packing_events(k, stage, n == 0, k_widths):
                required.append(scales[n - int(previous), record])
        budget = min((len(events) - 1 - index for index in required), default=63)
        result.append(budget)
        request(q + 2)
    return tuple(result)


def save_8x1_state(
    b_prefetch,
    packed,
    scales,
    addresses,
    *,
    c,
    ptpc,
    first_unpacked,
    prepare_b_addresses,
):
    # Keep FP8 payloads as integer bits across SCF loop boundaries.
    state = [
        b_prefetch[index].load().bitcast(fx.Uint32) for index in range_constexpr(2)
    ]
    for row in range_constexpr(2):
        for group in range_constexpr(2 * first_unpacked, 8):
            state.append(c[None, group, row].load())
    if const_expr(ptpc):
        for pair in range_constexpr(first_unpacked, 4):
            state.append(scales[pair].load())
    for pair in range_constexpr(first_unpacked):
        for row in range_constexpr(2):
            state.append(packed[pair][row])
    if const_expr(prepare_b_addresses is not None):
        state.extend(addresses)
    return state


def restore_8x1_state(
    state, *, c, ptpc, first_unpacked, prepare_b_addresses, b_carriers, scale_carriers
):
    for index in range_constexpr(2):
        b_carriers[index].store(state[index].bitcast(b_carriers[index].dtype))
    offset = 2
    scales = []
    for row in range_constexpr(2):
        for group in range_constexpr(2 * first_unpacked, 8):
            c[None, group, row].store(state[offset])
            offset += 1
    for pair in range_constexpr(first_unpacked, 4):
        if const_expr(ptpc):
            scale_carriers[pair - first_unpacked].store(state[offset])
            offset += 1
            scales.append(scale_carriers[pair - first_unpacked])
        else:
            scales.append(fx.Float32(1.0))
    packed = []
    for pair in range_constexpr(first_unpacked):
        packed.append([Vec(state[offset]), Vec(state[offset + 1])])
        offset += 2
    addresses = (
        [fx.Int32(value) for value in state[-4:]]
        if const_expr(prepare_b_addresses is not None)
        else []
    )
    return list(b_carriers), packed, scales, addresses


# ==================== K=256/384/512/640: BK128 multiples, two-slot weight pipeline ====================


@flyc.jit
def _emit_k128n_body(
    p_input,
    p_weight,
    p_output,
    p_sorted_ids,
    p_sorted_weights,
    p_sorted_expert_ids,
    p_w_scale,
    p_a_scale,
    M,
    e_idx,
    tid,
    lane,
    wave,
    group,
    group_tid,
    N,
    K,
    TOPK,
    tile_k,
    down_output_padding_bytes,
    weight_quant_type,
    act_quant_type,
    _task_table,
    _store_cache,
    ops,
):
    # Configuration and primitives.
    # Shapes and quantization are host-static; task mapping and thread coordinates pass through.
    BM = 256
    BN = 128
    BK = tile_k
    NUM_WAVES = 8
    WAVE_M = BM // NUM_WAVES
    KS = K // BK
    NT = N // BN
    K_WIDTHS = (BK,) * KS
    K_OFFSETS = tuple(ks * BK for ks in range(KS))
    FIRST_UNPACKED = 3
    LOOP_END = max(2, NT - 1)
    WEIGHT_QUARTER_ATOMS = BN * BK // (16 * 4)
    WEIGHT_PREFETCH_SLOTS = 2
    OUTPUT_STORES_PER_WAVE = WAVE_M * BN * 2 // (64 * 16)
    CSHUFFLE_N_PAIRS = BN // 32
    STRIDE = N + down_output_padding_bytes // 2
    use_n_loop = NT >= 3
    CSHUFFLE_WAVES = 4

    assert WEIGHT_QUARTER_ATOMS == 256
    assert OUTPUT_STORES_PER_WAVE == 8

    if const_expr(_task_table):
        row_begin = p_sorted_expert_ids[2 * e_idx]
    input_tensor = fx.rocdl.make_buffer_tensor(
        fxh.view_as_torch_tensor(p_input, (M, TOPK, K), fx.Float8E4M3FNUZ),
        max_size=False,
        num_records_bytes=fx.Int64(M) * TOPK * K,
    )
    sorted_ids = fx.rocdl.make_buffer_tensor(
        fxh.view_as_torch_tensor(
            fxh._as_ptr(p_sorted_ids)
            + (
                fx.Int64(row_begin) if const_expr(_task_table) else fx.Int64(e_idx) * BM
            ),
            (BM,),
            fx.Int32,
        ),
        max_size=False,
        num_records_bytes=BM * 4,
    )
    sorted_weights = fxh.view_as_torch_tensor(
        fxh._as_ptr(p_sorted_weights)
        + (fx.Int64(row_begin) if const_expr(_task_table) else fx.Int64(e_idx) * BM),
        (BM,),
        fx.Float32,
    )
    if const_expr(_task_table):
        expert_id = p_sorted_expert_ids[2 * e_idx + 1]
    else:
        expert_id = fxh.view_as_torch_tensor(p_sorted_expert_ids, (1,), fx.Int32)[e_idx]

    shared_allocator = fx.SharedAllocator()
    if const_expr(use_n_loop):
        weight_slots = shared_allocator.allocate(
            fx.Array[fx.Float8E4M3FNUZ, 2 * BN * BK, 16]
        )
        weight_storage_ptrs = [
            weight_slots.peek().ptr,
            weight_slots.peek().ptr + BN * BK,
        ]
    else:
        weight_ping_storage = shared_allocator.allocate(
            fx.Array[fx.Float8E4M3FNUZ, BN * BK, 16]
        )
        weight_pong_storage = shared_allocator.allocate(
            fx.Array[fx.Float8E4M3FNUZ, BN * BK, 16]
        )
        weight_storage_ptrs = [
            weight_ping_storage.peek().ptr,
            weight_pong_storage.peek().ptr,
        ]
    cshuffle_storage = shared_allocator.allocate(
        fx.Array[fx.BFloat16, CSHUFFLE_WAVES * 16 * BN, 16]
    )
    sorted_lds = fx.make_view(
        fx.recast_iter(fx.Int32, cshuffle_storage.peek().ptr),
        fx.make_layout(BM, 1),
    )
    if tid < BM:
        sorted_lds[tid] = sorted_ids[tid]
    gpu.barrier()

    mm = ops.create_thr_mma(fx.Float8E4M3FNUZ, (1, NUM_WAVES, 1))
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))
    c_fake = fx.make_view(
        fx.get_iter(input_tensor), fx.make_ordered_layout((BN, BM), (0, 1))
    )
    frag_c = mm.make_fragment_C(c_fake)
    row_tensor = fx.make_view(
        fx.get_iter(sorted_weights), fx.make_layout((BN, BM), (0, 1))
    )
    frag_row_scale = ops.load_tiled_mma_fragC(mm, row_tensor, copy_atom_bits=32)
    if const_expr(act_quant_type == "ptpc"):
        coord_tensor = fx.make_view(
            fx.get_iter(sorted_lds), fx.make_layout((BN, BM), (0, 1))
        )
        frag_coord = ops.load_tiled_mma_fragC(mm, coord_tensor, copy_atom_bits=32)
        a_scale_tensor = fx.rocdl.make_buffer_tensor(
            fxh.view_as_torch_tensor(p_a_scale, (M, TOPK), fx.Float32),
            max_size=False,
            num_records_bytes=fx.Int64(M) * TOPK * 4,
        )
        a_scale_copy = ops.get_buffer_copy_atom(fx.Float32, 32)
        frag_a_scale = mm.make_fragment_C(coord_tensor)
        frag_a_scale_retile = ops.get_tiled_mma_retile(
            mm, frag_a_scale, "C", copy_atom=a_scale_copy
        )
        for dst, coord in fxh.all_elements(frag_a_scale_retile, frag_coord):
            sorted_id = coord[0].bitcast(fx.Uint32)
            source = fxh.atom_tensor(
                a_scale_tensor, (sorted_id & 0xFFFFFF, sorted_id >> 24), 32
            )
            fx.copy(a_scale_copy, source, dst)
        frag_row_scale.store(frag_row_scale.load() * frag_a_scale.load())
        if const_expr(weight_quant_type == "per_tensor"):
            scalar_weight_scale = fx.make_view(
                fxh._as_ptr(p_w_scale, fx.Float32) + expert_id,
                fx.make_layout(1, 1),
            )[0]
            frag_row_scale.store(frag_row_scale.load() * scalar_weight_scale)
    else:
        # Fuse the per-expert weight scale and global activation scale in advance.
        scalar_a_scale = fx.make_view(
            fxh._as_ptr(p_a_scale, fx.Float32), fx.make_layout(1, 1)
        )[0]
        scalar_w_scale = fx.make_view(
            fxh._as_ptr(p_w_scale, fx.Float32) + expert_id,
            fx.make_layout(1, 1),
        )[0]
        frag_row_scale.store(frag_row_scale.load() * (scalar_a_scale * scalar_w_scale))

    weight_base = (
        fx.recast_iter(fx.Float8E4M3FNUZ, fxh._as_ptr(p_weight))
        + fx.Int64(expert_id) * N * K
    )
    weight_view = fx.make_view(weight_base, fx.make_layout(N * K, 1))
    weight_flat = fx.rocdl.make_buffer_tensor(
        weight_view, max_size=False, num_records_bytes=N * K
    )
    weight_packet = BufferTensor(
        fx.make_view(fx.get_iter(weight_flat), fx.make_layout(16, 1))
    )
    weight_staging = [
        fx.make_rmem_tensor(fx.make_layout(16, 1), fx.Float8E4M3FNUZ)
        for _ in range_constexpr(WEIGHT_PREFETCH_SLOTS)
    ]
    weight_store_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float8E4M3FNUZ)

    def weight_lds_quarter_view(pointer, n_quarter):
        return fx.make_view(
            pointer + n_quarter * (BN // 4) * BK,
            fx.make_layout(((16, BN // 64), (16, BK // 16)), ((16, 16 * BK), (1, 256))),
        )

    # Build whole-half layouts only here; retain the original first-tile reads for short N.
    lds_weight_halves = [
        [
            fx.make_view(
                storage + n_half * (BN // 2) * BK,
                fx.make_layout(
                    ((16, BN // 32), (16, BK // 16)), ((16, 16 * BK), (1, 256))
                ),
            )
            for n_half in range_constexpr(2)
        ]
        for storage in weight_storage_ptrs
    ]
    lds_weight_quarters = [
        [
            weight_lds_quarter_view(storage, n_quarter)
            for n_quarter in range_constexpr(4)
        ]
        for storage in weight_storage_ptrs
    ]

    quarter_atom_index = group * WEIGHT_QUARTER_ATOMS + group_tid
    quarter_n_group = quarter_atom_index // BK
    quarter_within_group = quarter_atom_index % BK
    quarter_k_group = quarter_within_group // 16
    quarter_n_inner = quarter_within_group % 16
    weight_quarter_lane_offset_bytes = (
        quarter_n_group * (16 * K) + quarter_k_group * 256 + quarter_n_inner * 16
    )

    def read_b_g2r(n, kb, half, *, prefetch_slot=0):
        core_base_bytes = n * (BN * K) + kb * (BK // 16) * 256 + half * (BN // 2) * K

        weight_staging[prefetch_slot].store(
            weight_packet.load(
                voffset_bytes=fx.Int32(weight_quarter_lane_offset_bytes),
                soffset_bytes=fx.Int32(core_base_bytes),
            )
        )
        return weight_staging[prefetch_slot]

    def store_b_r2s(
        lds_slot, kb, half, *, prefetch_slot=0, fragment=None, address=None
    ):
        if const_expr(fragment is not None):
            weight_staging[prefetch_slot].store(fragment.load())
        if const_expr(address is not None):
            destination = LdsTensor(
                fx.make_view(b_write_pointer, fx.make_layout(16, 1))
            )
            destination.store(
                weight_staging[prefetch_slot],
                address_bytes=address,
                offset_bytes=half * (BN // 2) * BK,
                copy_atom=weight_store_atom,
            )
        else:
            n_group = quarter_atom_index // BK
            within_group = quarter_atom_index % BK
            k_group = within_group // 16
            n_inner = within_group % 16
            lds_offset = (
                half * (BN // 2) * BK
                + n_group * (16 * BK)
                + k_group * 256
                + n_inner * 16
            )
            destination = fx.make_view(
                (
                    weight_storage_ptrs[0] + lds_slot * BN * BK
                    if const_expr(use_n_loop)
                    else weight_storage_ptrs[lds_slot]
                )
                + lds_offset,
                fx.make_layout(16, 1),
            )
            fx.copy(weight_store_atom, weight_staging[prefetch_slot], destination)

    def shuffle_c_r2s2r(block_n, packed_super_records, row_pair, n_half):
        packed_records = [
            packed_super_records[n_half * (CSHUFFLE_N_PAIRS // 2) + local_n_pair][
                row_pair
            ]
            for local_n_pair in range_constexpr(CSHUFFLE_N_PAIRS // 2)
        ]
        # Write both records in the original LDS planes, then read two 8B pieces per output row.
        for local_n_pair in range_constexpr(CSHUFFLE_N_PAIRS // 2):
            n_pair = n_half * (CSHUFFLE_N_PAIRS // 2) + local_n_pair
            lds_offset = wave_lds_base + cshuffle_plane_offset(
                lane_row, lane_group, n_pair
            )
            destination = fx.make_view(
                fx.get_iter(cshuffle_lds) + lds_offset, fx.make_layout(8, 1)
            )
            fragment = fx.make_fragment_like(destination)
            fragment.store(packed_records[local_n_pair])
            fx.copy(cshuffle_write_atom, fragment, destination)

        output_fragments, destinations = [], []
        for output_row_half in range_constexpr(2):
            fragment_pair = []
            for source_group in range_constexpr(2):
                source = fx.make_view(
                    cshuffle_read_pointers[2 * n_half + output_row_half]
                    + source_group * 1024,
                    fx.make_layout(4, 1),
                )
                fragment = fx.make_fragment_like(source)
                fx.copy(cshuffle_read_atom, source, fragment)
                fragment_pair.append(fragment)
            # Merge only the two 8B pieces of the same row; cross-row pairing adds moves.
            fx.rocdl.sched_barrier(0)
            output_fragments.append(fragment_pair)
            destination_index = n_half * 2 + output_row_half
            destinations.append(
                (
                    block_n,
                    fx.make_view(
                        fx.get_iter(output_tensor)
                        + output_destination_offsets[row_pair][destination_index],
                        fx.make_layout(8, 1),
                    ),
                )
            )
        return output_fragments, destinations

    def b_startup_coords(half_core):
        # Startup uses q=0/1 to prefetch Q1/Q2; at least four steps per N guarantee consumers.
        target = half_core + 1
        target_n = target // (2 * KS)
        target_k = target % KS
        return (
            target_n,
            target_k,
            (target % (2 * KS)) // KS,
            (target_n * KS + target_k) & 1,
        )

    def prepare_b_addresses(n):
        # Prepare both relative-slot read/write addresses at the previous compute tail; carry across N.
        parity = (n * KS) & 1
        read_base = fx.Int32(fx.ptrtoint(b_read_pointer))
        write_base = fx.Int32(fx.ptrtoint(b_write_pointer))
        values = [
            base + fx.Int32(((parity + relative_slot) & 1) * BN * BK)
            for base in (read_base, write_base)
            for relative_slot in range_constexpr(2)
        ]

        return [
            fx.Int32(
                llvm.inline_asm(
                    ir.IntegerType.get_signless(32),
                    [as_ir_value(value)],
                    "",
                    "=v,0",
                    has_side_effects=True,
                )
            )
            for value in values
        ]

    def read_b_s2r(lds_slot, kb, half, *, packet, address=None):
        if const_expr(packet is None):
            # N0 q0 and the remaining first-tile steps for short N retain whole-half read/register layouts.
            return ops.load_tiled_mma_fragA(
                mm, lds_weight_halves[lds_slot][half], copy_atom_bits=128
            )
        if const_expr(address is not None):
            source = LdsTensor(b_partition)
            fragment = mm.make_fragment_A(b_template)
            source.load(
                address_bytes=address,
                offset_bytes=(2 * half + packet) * (BN // 4) * BK,
                into=ops.get_tiled_mma_retile(mm, fragment, "A", copy_atom=b_copy),
                copy_atom=b_copy,
            )
            return fragment
        if const_expr(use_n_loop):
            view = weight_lds_quarter_view(
                weight_storage_ptrs[0] + lds_slot * BN * BK, 2 * half + packet
            )
        else:
            view = lds_weight_quarters[lds_slot][2 * half + packet]
        return ops.load_tiled_mma_fragA(mm, view, copy_atom_bits=128)

    # Closures bind views and resident A; SSA values changing across N still pass through explicit state.
    a_fragments = output_tensor = cshuffle_lds = None
    cshuffle_write_atom = cshuffle_read_atom = output_store_atom = None
    lane_group = lane_row = wave_lds_base = None
    output_destination_offsets = cshuffle_read_bases = cshuffle_read_pointers = (
        weight_scale_buffer
    ) = None
    b_template = b_copy = b_partition = b_read_pointer = b_write_pointer = None

    def prepare_views():
        nonlocal output_tensor, cshuffle_lds, cshuffle_write_atom, cshuffle_read_atom, output_store_atom
        nonlocal lane_group, lane_row, wave_lds_base, output_destination_offsets, cshuffle_read_bases, weight_scale_buffer
        nonlocal b_template, b_copy, b_partition, b_read_pointer, b_write_pointer
        if const_expr(prepare_addresses is not None):
            # Prepare the B template, partition, and base pointers here; pin N0 addresses after the prologue.
            b_template = weight_lds_quarter_view(weight_storage_ptrs[0], 0)
            b_copy = ops.get_universal_copy_atom(fx.Float8E4M3FNUZ, 128)
            b_partition = ops.get_tiled_mma_partition_S(
                mm, b_template, "A", copy_atom_bits=128
            )
            b_read_pointer = fx.get_iter(b_partition)
            b_write_pointer = (
                weight_storage_ptrs[0]
                + (quarter_atom_index // BK) * (16 * BK)
                + ((quarter_atom_index % BK) // 16) * 256
                + (quarter_atom_index % 16) * 16
            )

        output_base = fxh._as_ptr(p_output, fx.BFloat16) + (
            fx.Int64(row_begin) * STRIDE
            if const_expr(_task_table)
            else fx.Int64(e_idx) * BM * STRIDE
        )
        output_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(output_base, fx.make_layout((N, BM), (1, STRIDE))),
            max_size=False,
            num_records_bytes=BM * STRIDE * 2,
        )
        cshuffle_lds = cshuffle_storage.peek().view(
            fx.make_layout(CSHUFFLE_WAVES * 16 * BN, 1)
        )
        cshuffle_write_atom = ops.get_universal_copy_atom(fx.BFloat16, 128)
        cshuffle_read_atom = ops.get_universal_copy_atom(fx.BFloat16, 64)
        output_store_atom = fx.make_copy_atom(
            fx.rocdl.BufferCopy128b(cache_modifier=_store_cache), fx.BFloat16
        )
        lane_group = lane // 16
        lane_row = lane % 16
        local_wave = wave % CSHUFFLE_WAVES
        wave_lds_base = local_wave * (16 * BN)
        output_destination_offsets = []
        for row_pair in range_constexpr(WAVE_M // 16):
            row_pair_offsets = []
            for n_half in range_constexpr(2):
                for output_row_half in range_constexpr(2):
                    output_atom = n_half * 8 + lane % 8
                    output_row = (
                        wave * 16
                        + row_pair * (NUM_WAVES * 16)
                        + output_row_half * 8
                        + lane // 8
                    )
                    row_pair_offsets.append(
                        output_tensor.layout(output_atom * 8, output_row)
                    )
            output_destination_offsets.append(row_pair_offsets)

        # Prepare only C read pointers here; side-effecting pins still follow A loads.
        cshuffle_read_bases = []
        for n_half in range_constexpr(2):
            for output_row_half in range_constexpr(2):
                output_atom = n_half * 8 + lane % 8
                n_group = output_atom // 2
                offset = (
                    wave_lds_base
                    + cshuffle_plane_offset(
                        output_row_half * 8 + lane // 8,
                        (output_atom % 2) * 2,
                        n_group // 2,
                    )
                    + (n_group % 2) * 4
                )
                cshuffle_read_bases.append(fx.get_iter(cshuffle_lds) + offset)

        if const_expr(weight_quant_type == "ptpc"):
            weight_scale_buffer = fx.rocdl.make_buffer_tensor(
                fxh.view_as_torch_tensor(
                    fxh._as_ptr(p_w_scale, fx.Float32) + fx.Int64(expert_id) * N,
                    (N,),
                    fx.Float32,
                ),
                max_size=False,
                num_records_bytes=N * 4,
            )

    def pin_c_read_addresses():
        nonlocal cshuffle_read_pointers
        cshuffle_read_pointers = []

        for index in range_constexpr(4):
            pointer = cshuffle_read_bases[index]
            address = fx.Int32(
                llvm.inline_asm(
                    ir.IntegerType.get_signless(32),
                    [as_ir_value(fx.Int32(fx.ptrtoint(pointer)))],
                    "",
                    "=v,0",
                    has_side_effects=True,
                )
            )
            cshuffle_read_pointers.append(fx.inttoptr(pointer.type, address))

    def mma(weight, kb, record, *, full_half=False):
        return mfma_8x1_record(
            weight,
            kb,
            record,
            k_widths=K_WIDTHS,
            afragments=a_fragments,
            c=frag_c,
            atom=mma_atom,
            weight_n_group_begin=2 * (record % 2) if full_half else 0,
        )

    def read_scale_g2r(n, pair):
        return read_8x1_scale_g2r(
            n,
            pair,
            weight_quant_type=weight_quant_type,
            scale_buffer=weight_scale_buffer,
            mm=mm,
            ops=ops,
        )

    def store_c_r2g(fragments, destinations, lgkmcnt=0):
        return store_8x1_c_r2g(
            fragments, destinations, lgkmcnt=lgkmcnt, store_atom=output_store_atom
        )

    # Bind existing tensors and compile-time callbacks; share scale/pack/store, not K-specific B layouts.
    clear_super_record = partial(clear_8x1_record, c=frag_c)
    constrain_mfma_valu_packet = partial(
        schedule_k128_pack, weight_quant_type=weight_quant_type
    )
    pack = partial(
        pack_8x1_record,
        c=frag_c,
        row_scale=frag_row_scale,
        weight_quant_type=weight_quant_type,
    )
    store_c_tile_r2g = partial(
        store_8x1_c_tile_r2g, shuffle_c_r2s2r=shuffle_c_r2s2r, store_c_r2g=store_c_r2g
    )
    # KS=3/5 flips LDS slots across N; carry full addresses. Even KS and short N need no callback.
    prepare_addresses = (
        prepare_b_addresses if const_expr(use_n_loop and K in (384, 640)) else None
    )
    save_state = partial(
        save_8x1_state,
        c=frag_c,
        ptpc=weight_quant_type == "ptpc",
        first_unpacked=FIRST_UNPACKED,
        prepare_b_addresses=prepare_addresses,
    )
    run_tile = partial(
        run_8x1_tile,
        k=K,
        n_tiles=NT,
        ptpc=weight_quant_type == "ptpc",
        read_b_g2r=read_b_g2r,
        store_b_r2s=store_b_r2s,
        read_b_s2r=read_b_s2r,
        read_scale_g2r=read_scale_g2r,
        pack=pack,
        shuffle_c_r2s2r=shuffle_c_r2s2r,
        store_c_r2g=store_c_r2g,
        mma=mma,
        clear=clear_super_record,
        schedule_pack=constrain_mfma_valu_packet,
        prepare_b_addresses=prepare_addresses,
    )

    def prologue():
        nonlocal a_fragments
        prepare_views()
        input_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float8E4M3FNUZ)
        a_fragments = [
            mm.make_fragment_B(
                fx.make_view(
                    fx.get_iter(input_tensor),
                    fx.make_layout((BM, width), (1, BM)),
                )
            )
            for width in K_WIDTHS
        ]
        # B (n,kb,half) denotes N128 tile, K block, and N64 half (0=L/1=H); slot is the buffer slot.
        prefetch0_n, prefetch0_kb, prefetch0_half, _prefetch0_lds_slot = (
            b_startup_coords(0)
        )
        prefetch1_n, prefetch1_kb, prefetch1_half, _prefetch1_lds_slot = (
            b_startup_coords(1)
        )

        # Execute: first B -> A -> Q0 to LDS -> Q1/Q2; C address pins remain after A.
        read_b_g2r(0, 0, 0, prefetch_slot=0)
        fx.rocdl.sched_barrier(0)
        read_a_g2r(
            input_tensor,
            sorted_lds,
            a_fragments,
            input_copy,
            lane,
            wave,
            K_WIDTHS,
            K_OFFSETS,
        )
        pin_c_read_addresses()
        rocdl.s_waitcnt(vmcnt=4)
        store_b_r2s(0, 0, 0, prefetch_slot=0)
        fx.rocdl.sched_barrier(0)
        read_b_g2r(prefetch0_n, prefetch0_kb, prefetch0_half, prefetch_slot=0)
        read_b_g2r(prefetch1_n, prefetch1_kb, prefetch1_half, prefetch_slot=1)
        rocdl.s_waitcnt(vmcnt=1)
        # Stagger before q0; Q0 writes from both groups must first be ready in LDS.
        rocdl.s_waitcnt(lgkmcnt=0)
        stage_end()
        frag_c.fill(0)
        return weight_staging

    def prepare_loop_state(b_prefetch, packed, scales, b_addresses):
        # Create carriers only after N1 completes; do not extend cross-N SSA lifetimes earlier.
        b_carriers = [
            fx.make_fragment_like(b_prefetch[index]) for index in range_constexpr(2)
        ]
        scale_carriers = None
        if const_expr(weight_quant_type == "ptpc"):
            scale_carriers = [
                fx.make_fragment_like(scales[pair])
                for pair in range_constexpr(FIRST_UNPACKED, 4)
            ]
        restore_state = partial(
            restore_8x1_state,
            c=frag_c,
            ptpc=weight_quant_type == "ptpc",
            first_unpacked=FIRST_UNPACKED,
            prepare_b_addresses=prepare_addresses,
            b_carriers=b_carriers,
            scale_carriers=scale_carriers,
        )

        return save_state(b_prefetch, packed, scales, b_addresses), restore_state

    # Pipeline: prologue keeps A resident, LDS=Q0, P0/P1=Q1/Q2.
    b_prefetch = prologue()
    b_addresses = (
        prepare_b_addresses(0) if const_expr(prepare_addresses is not None) else []
    )

    # N0: stagger before q0; pack C0/C1/C2, leaving FP32 C3 for the next N.
    # Group1's wait pairs with group0's final q0 Memory barrier.
    if group == 1:
        stage_end()
    b_prefetch, c_bf16, c_scales, b_addresses = run_tile(
        0,
        b_prefetch,
        previous_packed=[],
        previous_scales=[],
        first=True,
        last=NT == 1,
        addresses=b_addresses,
    )

    # N1: finish packing/drain N0 while producing N1; pass only the unpacked C3's old scale.
    if const_expr(NT > 1):
        b_prefetch, c_bf16, c_scales, b_addresses = run_tile(
            1,
            b_prefetch,
            c_bf16,
            c_scales[FIRST_UNPACKED:],
            first=False,
            last=NT == 2,
            addresses=b_addresses,
        )

    # Loop: retire old C/compute new C with a fixed one-N backedge; NT=1/2 has no SSA, NT=3 stays zero-trip.
    if const_expr(NT >= 3):
        initial, restore_state = prepare_loop_state(
            b_prefetch, c_bf16, c_scales, b_addresses
        )
        for block_start, state in range(2, LOOP_END, 1, init=initial):
            b_prefetch, previous_packed, previous_scales, b_addresses = restore_state(
                state
            )
            b_prefetch, packed, scales, b_addresses = run_tile(
                fx.Int64(block_start) + 0,
                b_prefetch,
                previous_packed,
                previous_scales,
                first=False,
                last=False,
                addresses=b_addresses,
            )
            results = yield save_state(b_prefetch, packed, scales, b_addresses)
        b_prefetch, c_bf16, previous_scales, b_addresses = restore_state(results)
    else:
        previous_scales = c_scales[FIRST_UNPACKED:]

    # Tail: still interleave old/new C; prune only B without consumers. Empty for NT=1/2.
    for n in range_constexpr(LOOP_END, NT):
        b_prefetch, c_bf16, c_scales, b_addresses = run_tile(
            n,
            b_prefetch,
            c_bf16,
            previous_scales,
            first=False,
            last=n == NT - 1,
            addresses=b_addresses,
        )
        previous_scales = c_scales[FIRST_UNPACKED:]

    # Epilogue: wait, finish packing C3, store all four output quarters, and retain group0 compensation.
    rocdl.s_waitcnt(vmcnt=0)
    for pair in range_constexpr(FIRST_UNPACKED, 4):
        c_bf16.append(pack(pair, c_scales[pair]))
    store_c_tile_r2g(NT - 1, c_bf16)
    stage_end()
    if group == 0:
        stage_end()


# Single-step timeline: shared by K128n and K320.


def run_8x1_tile(
    n,
    b_prefetch,
    previous_packed,
    previous_scales,
    *,
    first=False,
    last=False,
    addresses,
    k,
    n_tiles,
    ptpc,
    read_b_g2r,
    store_b_r2s,
    read_b_s2r,
    read_scale_g2r,
    pack,
    shuffle_c_r2s2r,
    store_c_r2g,
    mma,
    clear,
    schedule_pack,
    k_widths=None,
    prepare_b_addresses=None,
):
    """q = 2*KS*n + step is only a time coordinate; n may be dynamic, but step is unrolled.

    Every N starts at step0; first=True marks only N0, with staggering already established.
    Complete previous_packed in place. The clear/mma/pack closures update FP32 c,
    which is not returned. packed/scales belong to the current N; scales returns
    all four records. previous_scales contains only the tail, indexed by record-FIRST_UNPACKED.
    K128n packs old C3 at step0 and new C0/C1/C2 at KS-1/KS/the final step.
    K320 packs old C2/C3 at step0 and new C0/C1 at step2.
    Only the rules in packing_events are executed below.
    """
    widths = k_widths or (128,) * (k // 128)
    KS = len(widths)
    budgets = vmem_wait_schedule(k, n_tiles, ptpc, k_widths)
    FIRST_UNPACKED = 2 if k == 320 else 3
    # Short-N wait rules apply to N0/N1 independently of whether this step reads a whole half.
    short_n_k128 = k != 320 and n_tiles < 3

    def b_target(n, step, *, ahead):
        target = step + ahead
        target_n, target_kb = n + target // (2 * KS), target % KS
        return (
            target_n,
            target_kb,
            (target % (2 * KS)) // KS,
            (target_n * KS + target_kb) & 1,
        )

    def has_b_target(step, *, ahead, last):
        return not last or step + ahead < 2 * KS

    packed, scales = [], []
    for step in range_constexpr(2 * KS):
        # kb indexes K blocks: 128 each for K128n, 128/192 for K320; it is not an offset or slot.
        # half=0/1 selects the first/last 64 columns of N128; visit all L K blocks before H.
        kb, half, prefetch_slot = step % KS, step // KS, step & 1
        lds_slot = (n * KS + kb) & 1
        # BK128 q0 always reads N64; so do remaining first-tile steps for short N. K320 reads two N32s.
        read_full_half = first and k != 320 and (step == 0 or short_n_k128)
        output = output_quarter(k, step, k_widths)
        has_output = not first and output is not None
        events = packing_events(k, step, first=first, k_widths=k_widths)

        # Memory: read Q[q], interleaving previous-N C stores; P and LDS slots rotate independently.
        priority(0)
        if const_expr(step < 4):
            scales.append(read_scale_g2r(n, step))
        if const_expr(has_output):
            # output encodes (row, half), not the N32 record used by Compute.
            fragments, destinations = shuffle_c_r2s2r(
                n - 1, previous_packed, output % 2, output // 2
            )
        b_read_address = (
            addresses[kb & 1] if const_expr(prepare_b_addresses is not None) else None
        )
        if const_expr(read_full_half):
            # Whole-half and two-packet reads are exclusive and use their matching MFMA indices.
            b_full = read_b_s2r(lds_slot, kb, half, packet=None, address=b_read_address)
        else:
            b0 = read_b_s2r(lds_slot, kb, half, packet=0, address=b_read_address)
            if const_expr(has_output):
                # Issue stores once old CShuffle reads complete; current B reads may remain in flight.
                store_c_r2g(fragments, destinations, lgkmcnt=widths[kb] // 32)
                fx.rocdl.sched_barrier(0)
            b1 = read_b_s2r(lds_slot, kb, half, packet=1, address=b_read_address)

        # Protect the next B submission and this step's pack scales; N1 is separate, the backedge uses n>=2.
        budget_n = (
            0 if first else n_tiles - 1 if last else n if isinstance(n, int) else 2
        )
        # For the final short-N step, lowering inserts the wait at scale use, not earlier in Memory.
        if const_expr(
            has_b_target(step, ahead=1, last=last)
            or (ptpc and not short_n_k128 and events)
        ):
            rocdl.s_waitcnt(vmcnt=budgets[budget_n * 2 * KS + step])
        if const_expr(has_b_target(step, ahead=1, last=last)):
            # B r->s submits Q[q+1]; materialize the target after waiting, consuming P[prefetch_slot].
            _b_r2s_n, b_r2s_kb, b_r2s_half, b_r2s_lds_slot = b_target(n, step, ahead=1)
            if const_expr(prepare_b_addresses is not None):
                # The last L K block advances to the same N's H/K0; odd KS cannot use current LDS slot+1.
                relative_slot = (b_r2s_kb + ((step + 1) // (2 * KS)) * KS) & 1
                write_address = addresses[2 + relative_slot]
            else:
                write_address = None
            store_b_r2s(
                b_r2s_lds_slot,
                b_r2s_kb,
                b_r2s_half,
                prefetch_slot=prefetch_slot,
                fragment=b_prefetch[prefetch_slot],
                address=write_address,
            )
        if const_expr(has_b_target(step, ahead=3, last=last)):
            # Prefetch Q[q+3] into the submitted P slot; materialize its target before the original sched barrier.
            b_g2r_n, b_g2r_kb, b_g2r_half, _b_g2r_lds_slot = b_target(n, step, ahead=3)
            fx.rocdl.sched_barrier(0)
            b_prefetch[prefetch_slot] = read_b_g2r(
                b_g2r_n,
                b_g2r_kb,
                b_g2r_half,
                prefetch_slot=prefetch_slot,
            )
        if const_expr(first and (step == 0 or (k == 320 and step == 1))):
            # Q1's cross-group consumers need both LDS writes; K320 also keeps its first H/K128 handoff.
            rocdl.s_waitcnt(lgkmcnt=0)
        stage_end()

        # Compute: preserve packet order, interleaving current-record MFMA with old/new-record packing.
        priority(3)
        # packet=0/1 selects each half's first/last 32 columns; record=0..3 identifies N32 outputs.
        for packet in range_constexpr(2):
            record = 2 * half + packet
            if const_expr(kb == 0):
                clear(record)
            fx.rocdl.sched_barrier(0)
            if const_expr(read_full_half):
                mma(b_full, kb, record, full_half=True)
            else:
                b_packet = b0 if const_expr(packet == 0) else b1
                mma(b_packet, kb, record)
            for pack_packet, from_previous_n, pack_record in events:
                if const_expr(packet == pack_packet):
                    if const_expr(from_previous_n):
                        previous_packed.append(
                            pack(
                                pack_record,
                                previous_scales[pack_record - FIRST_UNPACKED],
                            )
                        )
                    else:
                        packed.append(pack(pack_record, scales[pack_record]))
                    schedule_pack()
        if const_expr(
            prepare_b_addresses is not None and step == 2 * KS - 1 and not last
        ):
            addresses = prepare_b_addresses(n + 1)
        rocdl.s_waitcnt(lgkmcnt=0)
        priority(0)
        stage_end()
    return b_prefetch, packed, scales, addresses


# ==================== K=192: whole-K192 two-slot pipeline, 48 MFMA per half/wave ====================


@cache
def k192_wait_schedule(n_tiles, ptpc):
    """Each FIFO half-B uses two VMEM instructions (16B+8B); protect B and cross-half pack scales."""
    sequence, requests, scales, budgets = 0, {}, {}, []

    def request_b_g2r(q):
        nonlocal sequence
        if q < 2 * n_tiles:
            sequence += 2
            requests[q] = sequence

    request_b_g2r(1)
    request_b_g2r(2)
    for n in range(n_tiles):
        for half in range(2):
            q = 2 * n + half
            if ptpc:
                for pair in range(2 * half, 2 * half + 2):
                    sequence += 2
                    scales[n, pair] = sequence
            if n > 0:
                sequence += 4
            required = [requests[q + 1]] if q + 1 < 2 * n_tiles else []
            if ptpc:
                if half == 0 and n > 0:
                    required.extend(scales[n - 1, pair] for pair in (2, 3))
                elif half == 1:
                    required.extend(scales[n, pair] for pair in (0, 1))
            budgets.append(min((sequence - event for event in required), default=63))
            request_b_g2r(q + 3)
    return tuple(budgets)


def save_k192_state(b_prefetch, packed, scales, addresses, *, c, ptpc):
    state = [part.load() for carry in b_prefetch for part in carry]
    for row in range_constexpr(2):
        for group in range_constexpr(4, 8):
            state.append(c[None, group, row].load())
    if const_expr(ptpc):
        state.extend(scales[pair].load() for pair in range_constexpr(2, 4))
    for pair in range_constexpr(2):
        state.extend(packed[pair][row] for row in range_constexpr(2))
    state.extend(addresses)
    return state


def restore_k192_state(state, *, c, ptpc, b_carriers, scale_carriers):
    for half in range_constexpr(2):
        for part in range_constexpr(2):
            b_carriers[half][part].store(state[half * 2 + part])
    offset, scales = 4, []
    for row in range_constexpr(2):
        for group in range_constexpr(4, 8):
            c[None, group, row].store(state[offset])
            offset += 1
    for pair in range_constexpr(2):
        if const_expr(ptpc):
            scale_carriers[pair].store(state[offset])
            scales.append(scale_carriers[pair])
            offset += 1
        else:
            scales.append(fx.Float32(1.0))
    packed = []
    for pair in range_constexpr(2):
        packed.append([Vec(state[offset]), Vec(state[offset + 1])])
        offset += 2
    return (
        [list(carry) for carry in b_carriers],
        packed,
        scales,
        [fx.Int32(value) for value in state[-5:]],
    )


@flyc.jit
def _emit_k192_body(
    p_input,
    p_weight,
    p_output,
    p_sorted_ids,
    p_sorted_weights,
    p_sorted_expert_ids,
    p_w_scale,
    p_a_scale,
    M,
    e_idx,
    tid,
    lane,
    wave,
    group,
    N,
    TOPK,
    padding,
    weight_quant_type,
    act_quant_type,
    _task_table,
    _store_cache,
    ops,
):
    # Configuration and primitives.
    # Derive only host-static shapes; task mapping and dynamic thread coordinates pass through.
    K, BM, BN = 192, 256, 128
    K_WIDTHS = (192,)
    K_OFFSETS = (0,)
    NT = N // BN
    STRIDE = N + padding // 2
    FIRST_UNPACKED = 2
    LOOP_END = max(2, NT - 2)

    def schedule_pack():
        # PTPC has 40 VALU instructions, per-tensor 24; each K192 packet has 24 MFMA instructions.
        for index in range_constexpr(24):
            rocdl.sched_group_barrier(0x8, 1, 0)
            rocdl.sched_group_barrier(
                0x2, 2 if weight_quant_type == "ptpc" and index < 16 else 1, 0
            )
        rocdl.sched_barrier(0)

    if const_expr(_task_table):
        row_begin = p_sorted_expert_ids[2 * e_idx]
    allocator = fx.SharedAllocator()
    bslots = allocator.allocate(fx.Array[fx.Float8E4M3FNUZ, 2 * BN * K, 16])
    # Both slots remain contiguous; precomputed addresses include slot offsets.
    bptr = bslots.peek().ptr
    scratch = allocator.allocate(fx.Array[fx.BFloat16, 4 * 16 * BN, 16])
    scratch_view = scratch.peek().view(fx.make_layout(4 * 16 * BN, 1))
    ids_lds = fx.make_view(
        fx.recast_iter(fx.Int32, scratch.peek().ptr), fx.make_layout(BM, 1)
    )
    ids = fx.rocdl.make_buffer_tensor(
        fxh.view_as_torch_tensor(
            fxh._as_ptr(p_sorted_ids)
            + (
                fx.Int64(row_begin) if const_expr(_task_table) else fx.Int64(e_idx) * BM
            ),
            (BM,),
            fx.Int32,
        ),
        max_size=False,
        num_records_bytes=BM * 4,
    )
    if tid < BM:
        ids_lds[tid] = ids[tid]
    gpu.barrier()

    a = fx.rocdl.make_buffer_tensor(
        fxh.view_as_torch_tensor(p_input, (M, TOPK, K), fx.Float8E4M3FNUZ),
        max_size=False,
        num_records_bytes=fx.Int64(M) * TOPK * K,
    )
    expert = (
        p_sorted_expert_ids[2 * e_idx + 1]
        if const_expr(_task_table)
        else fxh.view_as_torch_tensor(p_sorted_expert_ids, (1,), fx.Int32)[e_idx]
    )
    weights = fx.rocdl.make_buffer_tensor(
        fx.make_view(
            fx.recast_iter(fx.Float8E4M3FNUZ, fxh._as_ptr(p_weight))
            + fx.Int64(expert) * N * K,
            fx.make_layout(N * K, 1),
        ),
        max_size=False,
        num_records_bytes=N * K,
    )
    mm = ops.create_thr_mma(fx.Float8E4M3FNUZ, (1, 8, 1))
    atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))
    c = mm.make_fragment_C(
        fx.make_view(fx.get_iter(a), fx.make_ordered_layout((BN, BM), (0, 1)))
    )
    routing_scale = fxh.view_as_torch_tensor(
        fxh._as_ptr(p_sorted_weights)
        + (fx.Int64(row_begin) if const_expr(_task_table) else fx.Int64(e_idx) * BM),
        (BM,),
        fx.Float32,
    )
    row_tensor = fx.make_view(
        fx.get_iter(routing_scale), fx.make_layout((BN, BM), (0, 1))
    )
    row_scale = ops.load_tiled_mma_fragC(mm, row_tensor, copy_atom_bits=32)
    if const_expr(act_quant_type == "ptpc"):
        coords = ops.load_tiled_mma_fragC(
            mm,
            fx.make_view(fx.get_iter(ids_lds), fx.make_layout((BN, BM), (0, 1))),
            copy_atom_bits=32,
        )
        als = fx.rocdl.make_buffer_tensor(
            fxh.view_as_torch_tensor(p_a_scale, (M, TOPK), fx.Float32),
            max_size=False,
            num_records_bytes=fx.Int64(M) * TOPK * 4,
        )
        scale_copy = ops.get_buffer_copy_atom(fx.Float32, 32)
        ascale = mm.make_fragment_C(row_tensor)
        for dst, coord in fxh.all_elements(
            ops.get_tiled_mma_retile(mm, ascale, "C", copy_atom=scale_copy), coords
        ):
            encoded = coord[0].bitcast(fx.Uint32)
            fx.copy(
                scale_copy,
                fxh.atom_tensor(als, (encoded & 0xFFFFFF, encoded >> 24), 32),
                dst,
            )
        row_scale.store(row_scale.load() * ascale.load())
        if const_expr(weight_quant_type == "per_tensor"):
            ws = fx.make_view(
                fxh._as_ptr(p_w_scale, fx.Float32) + expert, fx.make_layout(1, 1)
            )[0]
            row_scale.store(row_scale.load() * ws)
    else:
        # Multiply global activation and per-expert weight scales into the routing scale in advance.
        scalar_a = fx.make_view(
            fxh._as_ptr(p_a_scale, fx.Float32), fx.make_layout(1, 1)
        )[0]
        scalar_w = fx.make_view(
            fxh._as_ptr(p_w_scale, fx.Float32) + expert, fx.make_layout(1, 1)
        )[0]
        row_scale.store(row_scale.load() * (scalar_a * scalar_w))

    read_b_packet_g2r = partial(read_b_g2r_packet, weights)
    # Closures bind views and resident A; SSA values changing across N still pass through explicit state.
    afragments = b_template = b_copy = b_partition = b_write_templates = None
    out = store_atom = scratch_write = scratch_read = None
    scratch_base = lane_group = scale_buffer = None

    def read_b_g2r(n, half):
        base = n * BN * K + half * (BN // 2) * K

        return [
            read_b_packet_g2r(tid * 16, base, 4),
            read_b_packet_g2r(8192 + tid * 8, base, 2),
        ]

    def prepare_b_addresses(n):
        read_address = fx.Int32(fx.ptrtoint(fx.get_iter(b_partition))) + fx.Int32(
            (n & 1) * BN * K
        )
        # Read L/write current-N H; read H/write next-N L. Prepare both full lane pointers here.
        write_base = fx.Int32(fx.ptrtoint(bptr))
        write_h = write_base + fx.Int32((n & 1) * BN * K + (BN // 2) * K)
        write_l = write_base + fx.Int32(((n + 1) & 1) * BN * K)
        values = [
            read_address,
            write_h + tid * 16,
            write_h + 8192 + tid * 8,
            write_l + tid * 16,
            write_l + 8192 + tid * 8,
        ]

        # Empty tied asm emits no ISA but forces addresses ready before the compute-tail sched barrier.
        return [
            fx.Int32(
                llvm.inline_asm(
                    ir.IntegerType.get_signless(32),
                    [as_ir_value(value)],
                    "",
                    "=v,0",
                    has_side_effects=True,
                )
            )
            for value in values
        ]

    def store_b_r2s(half, fragments, *, addresses):
        for part in range_constexpr(2):
            words = 4 if part == 0 else 2
            destination = LdsTensor(
                b_write_templates[part]
            )  # pyright: ignore[reportOptionalSubscript]
            destination.store(
                fragments[part],
                address_bytes=addresses[1 + (1 - half) * 2 + part],
                copy_atom=ops.get_universal_copy_atom(fx.Uint32, words * 32),
            )

    def read_b_s2r(half, *, packet, addresses):
        source = LdsTensor(b_partition)
        fragment = mm.make_fragment_A(b_template)

        source.load(
            address_bytes=addresses[0],
            offset_bytes=half * (BN // 2) * K + packet * (BN // 4) * K,
            into=ops.get_tiled_mma_retile(mm, fragment, "A", copy_atom=b_copy),
            copy_atom=b_copy,
        )
        return fragment

    def prepare_views():
        nonlocal b_template, b_copy, b_partition, b_write_templates
        nonlocal out, store_atom, scratch_write, scratch_read, scratch_base, lane_group, scale_buffer
        # Build the partition once; full lane addresses are prepared at the previous compute tail and carried.
        b_template = fx.make_view(
            bptr,
            fx.make_layout(((16, BN // 64), (16, K // 16)), ((16, 16 * K), (1, 256))),
        )
        b_copy = ops.get_universal_copy_atom(fx.Float8E4M3FNUZ, 128)
        b_partition = ops.get_tiled_mma_partition_S(
            mm, b_template, "A", copy_atom_bits=128
        )
        b_write_ptr = fx.recast_iter(fx.Uint32, bptr)
        # The Uint32 templates preserve the pointer types/alignments of b_write_ptr and b_write_ptr+2.
        b_write_templates = [
            fx.make_view(b_write_ptr, fx.make_layout(4, 1)),
            fx.make_view(b_write_ptr + 2, fx.make_layout(2, 1)),
        ]

        out = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fxh._as_ptr(p_output, fx.BFloat16)
                + (
                    fx.Int64(row_begin) * STRIDE
                    if const_expr(_task_table)
                    else fx.Int64(e_idx) * BM * STRIDE
                ),
                fx.make_layout((N, BM), (1, STRIDE)),
            ),
            max_size=False,
            num_records_bytes=BM * STRIDE * 2,
        )
        store_atom = fx.make_copy_atom(
            fx.rocdl.BufferCopy128b(cache_modifier=_store_cache), fx.BFloat16
        )
        scratch_write = ops.get_universal_copy_atom(fx.BFloat16, 128)
        scratch_read = ops.get_universal_copy_atom(fx.BFloat16, 64)
        scratch_base = (wave % 4) * 16 * BN
        lane_group = lane // 16

        if const_expr(weight_quant_type == "ptpc"):
            scale_buffer = fx.rocdl.make_buffer_tensor(
                fxh.view_as_torch_tensor(
                    fxh._as_ptr(p_w_scale, fx.Float32) + fx.Int64(expert) * N,
                    (N,),
                    fx.Float32,
                ),
                max_size=False,
                num_records_bytes=N * 4,
            )

    def read_scale_g2r(n, pair):
        return read_8x1_scale_g2r(
            n,
            pair,
            weight_quant_type=weight_quant_type,
            scale_buffer=(
                scale_buffer if const_expr(weight_quant_type == "ptpc") else None
            ),
            mm=mm,
            ops=ops,
        )

    def mma(weight, ks, pair):
        return mfma_8x1_record(
            weight, ks, pair, k_widths=K_WIDTHS, afragments=afragments, c=c, atom=atom
        )

    def shuffle_c_r2s2r(n, packed, row, half):
        return shuffle_8x1_c_r2s2r(
            n,
            packed,
            row,
            half,
            scratch_view=scratch_view,
            scratch_base=scratch_base,
            lane=lane,
            lane_group=lane_group,
            wave=wave,
            out=out,
            scratch_write=scratch_write,
            scratch_read=scratch_read,
        )

    def store_c_r2g(fragments, destinations, lgkmcnt=0):
        return store_8x1_c_r2g(
            fragments, destinations, lgkmcnt=lgkmcnt, store_atom=store_atom
        )

    pack = partial(
        pack_8x1_record, c=c, row_scale=row_scale, weight_quant_type=weight_quant_type
    )
    store_c_tile_r2g = partial(
        store_8x1_c_tile_r2g, shuffle_c_r2s2r=shuffle_c_r2s2r, store_c_r2g=store_c_r2g
    )
    clear = partial(clear_8x1_record, c=c)
    save_state = partial(save_k192_state, c=c, ptpc=weight_quant_type == "ptpc")
    run_tile = partial(
        run_k192_tile,
        n_tiles=NT,
        ptpc=weight_quant_type == "ptpc",
        read_b_g2r=read_b_g2r,
        store_b_r2s=store_b_r2s,
        read_b_s2r=read_b_s2r,
        read_scale_g2r=read_scale_g2r,
        pack=pack,
        shuffle_c_r2s2r=shuffle_c_r2s2r,
        store_c_r2g=store_c_r2g,
        mma=mma,
        clear=clear,
        schedule_pack=schedule_pack,
        prepare_b_addresses=prepare_b_addresses,
    )

    def prologue():
        nonlocal afragments
        prepare_views()
        acopy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float8E4M3FNUZ)
        afragments = [
            mm.make_fragment_B(
                fx.make_view(
                    fx.get_iter(a),
                    fx.make_layout((BM, width), (1, BM)),
                )
            )
            for width in K_WIDTHS
        ]
        q0_destinations, q0_copies = [], []
        for part in range_constexpr(2):
            words = 4 if part == 0 else 2
            offset = tid * 16 if part == 0 else 8192 + tid * 8
            q0_destinations.append(
                fx.make_view(
                    fx.recast_iter(fx.Uint32, bptr) + offset // 4,
                    fx.make_layout(words, 1),
                )
            )
            q0_copies.append(ops.get_universal_copy_atom(fx.Uint32, words * 32))

        # Execute: first B -> A -> Q0 to LDS -> Q1/Q2; Q0 stays 16B+8B, A loads only three K64s.
        first_b = read_b_g2r(0, 0)
        rocdl.sched_barrier(0)
        read_a_g2r(a, ids_lds, afragments, acopy, lane, wave, K_WIDTHS, K_OFFSETS)
        rocdl.s_waitcnt(vmcnt=0)
        for part in range_constexpr(2):
            fx.copy(q0_copies[part], first_b[part], q0_destinations[part])
        rocdl.s_waitcnt(lgkmcnt=0)
        stage_end()
        c.fill(0)
        b_prefetch = [read_b_g2r(0, 1), None]  # P0=Q1=current H.
        if const_expr(NT > 1):
            b_prefetch[1] = read_b_g2r(1, 0)  # P1=Q2=next N's L.
        return b_prefetch

    def prepare_loop_state(b_prefetch, packed, scales, b_addresses):
        # Create original backedge carriers only after N1; preserve order and initial save timing.
        b_carriers = [
            [fx.make_fragment_like(part) for part in carry] for carry in b_prefetch
        ]
        scale_carriers = None
        if const_expr(weight_quant_type == "ptpc"):
            scale_carriers = [
                fx.make_fragment_like(scales[pair])
                for pair in range_constexpr(FIRST_UNPACKED, 4)
            ]
        restore_state = partial(
            restore_k192_state,
            c=c,
            ptpc=weight_quant_type == "ptpc",
            b_carriers=b_carriers,
            scale_carriers=scale_carriers,
        )

        return save_state(b_prefetch, packed, scales, b_addresses), restore_state

    # Pipeline: prologue keeps A resident, LDS=Q0, P0/P1=Q1/Q2; no prefetch without consumers.
    b_prefetch = prologue()
    b_addresses = prepare_b_addresses(0)

    # N0: stagger before q0's L/H steps; pack C0/C1, leaving FP32 C2/C3 for the next N.
    if group == 1:
        stage_end()
    b_prefetch, c_bf16, c_scales, b_addresses = run_tile(
        0,
        b_prefetch,
        previous_packed=[],
        previous_scales=[],
        first=True,
        last=NT == 1,
        addresses=b_addresses,
    )

    # N1: finish packing/drain N0 while producing N1; use the first-transition budget only once.
    if const_expr(NT > 1):
        b_prefetch, c_bf16, c_scales, b_addresses = run_tile(
            1,
            b_prefetch,
            c_bf16,
            c_scales[FIRST_UNPACKED:],
            first=False,
            last=NT == 2,
            addresses=b_addresses,
        )

    # Loop: retire old C/compute new C; dynamic n+2 is always valid, NT=4 keeps a zero-trip backedge.
    if const_expr(NT >= 4):
        initial, restore_state = prepare_loop_state(
            b_prefetch, c_bf16, c_scales, b_addresses
        )
        for block_start, state in range(2, LOOP_END, 1, init=initial):
            b_prefetch, previous_packed, previous_scales, b_addresses = restore_state(
                state
            )
            b_prefetch, packed, scales, b_addresses = run_tile(
                fx.Int64(block_start) + 0,
                b_prefetch,
                previous_packed,
                previous_scales,
                first=False,
                last=False,
                addresses=b_addresses,
            )
            results = yield save_state(b_prefetch, packed, scales, b_addresses)
        b_prefetch, c_bf16, previous_scales, b_addresses = restore_state(results)
    else:
        previous_scales = c_scales[FIRST_UNPACKED:]

    # Tail: the last two N still interleave old/new C; prune half-prefetches by actual consumers.
    for n in range_constexpr(LOOP_END, NT):
        b_prefetch, c_bf16, c_scales, b_addresses = run_tile(
            n,
            b_prefetch,
            c_bf16,
            previous_scales,
            first=False,
            last=n == NT - 1,
            addresses=b_addresses,
        )
        previous_scales = c_scales[FIRST_UNPACKED:]

    # Epilogue: wait, finish packing C2/C3 with the final tile's full scales, then drain C stores.
    rocdl.s_waitcnt(vmcnt=0)
    for pair in range_constexpr(FIRST_UNPACKED, 4):
        c_bf16.append(pack(pair, c_scales[pair]))
    store_c_tile_r2g(NT - 1, c_bf16)
    stage_end()
    if group == 0:
        stage_end()


# Single-step timeline: K192's independent L/H steps.


def run_k192_tile(
    n,
    b_prefetch,
    previous_packed,
    previous_scales,
    *,
    first=False,
    last=False,
    addresses,
    n_tiles,
    ptpc,
    read_b_g2r,
    store_b_r2s,
    read_b_s2r,
    read_scale_g2r,
    pack,
    shuffle_c_r2s2r,
    store_c_r2g,
    mma,
    clear,
    schedule_pack,
    prepare_b_addresses,
):
    """q = 2*n + half is only a time label; stagger before N0, then run L/H for every N.

    Each P slot holds 16B+8B; addresses holds current-N reads and current-N.H/next-N.L writes.
    Complete old C2/C3 in previous_packed in place; closures update FP32 c. Return
    current-N packed C0/C1 and all four scale records; advance addresses unless last.
    previous_scales contains only old scale2/scale3. Prune the final two N independently,
    without using the shared BK128 tail.
    """
    budgets = k192_wait_schedule(n_tiles, ptpc)
    body_budgets = tuple(
        min((budgets[2 * n + half] for n in range(2, n_tiles - 2)), default=63)
        for half in range(2)
    )
    has_next = not last
    # Dynamic n occurs only before the last two N; do not convert runtime SSA to a host bool.
    has_future = n + 2 < n_tiles if isinstance(n, int) else True
    packed, scales = [], []

    # K192 has one K block (kb=0); half=0/1 processes the first/last 64 columns of current N128.
    for half in range_constexpr(2):
        prefetch_slot = half
        # Memory: read Q[2*n+half], interleaving previous-N CShuffle/stores per packet.
        priority(0)
        for pair in range_constexpr(2 * half, 2 * half + 2):
            scales.append(read_scale_g2r(n, pair))
        b_packets = []
        for packet in range_constexpr(2):
            if const_expr(not first):
                # C output quarters encode (row=packet, half), not the N32 pack_record.
                fragments, destinations = shuffle_c_r2s2r(
                    n - 1, previous_packed, packet, half
                )
            b_packets.append(read_b_s2r(half, packet=packet, addresses=addresses))
            if const_expr(not first):
                # C reads precede the next six B ds_reads; issue stores without waiting for all of B.
                store_c_r2g(fragments, destinations, lgkmcnt=6)
                fx.rocdl.sched_barrier(0)
        budget = budgets[2 * n + half] if isinstance(n, int) else body_budgets[half]
        # Protect the next B submission and this step's pack scales; later prefetches do not count.
        if const_expr(budget != 63):
            rocdl.s_waitcnt(vmcnt=budget)
        if const_expr(half == 0 or has_next):
            # B r->s submits Q[q+1]: L writes current-N.H, H writes next-N.L.
            store_b_r2s(1 - half, b_prefetch[prefetch_slot], addresses=addresses)
        if const_expr((half == 0 and has_next) or (half == 1 and has_future)):
            # Prefetch Q[q+3] into just-submitted P[half]: next-N.H or following-N.L.
            fx.rocdl.sched_barrier(0)
            b_prefetch[prefetch_slot] = read_b_g2r(n + 1 + half, 1 - half)
        if const_expr(first and half == 0):
            # Already staggered before q0; the other group must see both groups' full Q1 before H.
            rocdl.s_waitcnt(lgkmcnt=0)
        stage_end()

        # Compute: retain packing interleaving and preparation of five next-N addresses.
        priority(3)
        # packet=0/1 selects the first/last 32 columns of the half; record=0..3 identifies N32 outputs.
        for packet in range_constexpr(2):
            record = half * 2 + packet
            clear(record)
            fx.rocdl.sched_barrier(0)
            mma(b_packets[packet], 0, record)
            if const_expr(half == 0 and not first):
                # After L computes new C0/C1, pack old C2/C3 with tail scales indexed by packet.
                pack_record = 2 + packet
                previous_packed.append(pack(pack_record, previous_scales[packet]))
                schedule_pack()
            elif const_expr(half == 1):
                # After H computes new C2/C3, pack this N's completed C0/C1.
                pack_record = packet
                packed.append(pack(pack_record, scales[pack_record]))
                schedule_pack()
        if const_expr(half == 1 and has_next):
            addresses = prepare_b_addresses(n + 1)
        rocdl.s_waitcnt(lgkmcnt=0)
        priority(0)
        stage_end()
    return b_prefetch, packed, scales, addresses


# ==================== K=320: four 128+192 stages, no padding or extra MFMA ====================


@flyc.jit
def _emit_k320_body(
    p_input,
    p_weight,
    p_output,
    p_sorted_ids,
    p_sorted_weights,
    p_sorted_expert_ids,
    p_w_scale,
    p_a_scale,
    M,
    e_idx,
    tid,
    lane,
    wave,
    group,
    group_tid,
    N,
    TOPK,
    padding,
    weight_quant_type,
    act_quant_type,
    _task_table,
    _store_cache,
    ops,
):
    # Configuration and primitives.
    # Preserve Python shape/wait-accounting expressions; task mapping and thread coordinates pass through.
    K, BM, BN = 320, 256, 128
    K_WIDTHS = (128, 192)
    K_OFFSETS = (0, 128)
    KS, NT = len(K_WIDTHS), N // BN
    STRIDE = N + padding // 2
    FIRST_UNPACKED = 2
    LOOP_END = max(2, NT - 1)

    if const_expr(_task_table):
        row_begin = p_sorted_expert_ids[2 * e_idx]
    allocator = fx.SharedAllocator()
    # KS=2 gives slot=(n*2+ks)&1=ks: fixed asymmetric 16/24KiB slots, 56KiB total LDS.
    b0 = allocator.allocate(fx.Array[fx.Float8E4M3FNUZ, BN * K_WIDTHS[0], 16])
    b1 = allocator.allocate(fx.Array[fx.Float8E4M3FNUZ, BN * K_WIDTHS[1], 16])
    bptrs = [b0.peek().ptr, b1.peek().ptr]
    scratch = allocator.allocate(fx.Array[fx.BFloat16, 4 * 16 * BN, 16])
    scratch_view = scratch.peek().view(fx.make_layout(4 * 16 * BN, 1))
    ids_lds = fx.make_view(
        fx.recast_iter(fx.Int32, scratch.peek().ptr), fx.make_layout(BM, 1)
    )
    ids = fx.rocdl.make_buffer_tensor(
        fxh.view_as_torch_tensor(
            fxh._as_ptr(p_sorted_ids)
            + (
                fx.Int64(row_begin) if const_expr(_task_table) else fx.Int64(e_idx) * BM
            ),
            (BM,),
            fx.Int32,
        ),
        max_size=False,
        num_records_bytes=BM * 4,
    )
    if tid < BM:
        ids_lds[tid] = ids[tid]
    gpu.barrier()

    a = fx.rocdl.make_buffer_tensor(
        fxh.view_as_torch_tensor(p_input, (M, TOPK, K), fx.Float8E4M3FNUZ),
        max_size=False,
        num_records_bytes=fx.Int64(M) * TOPK * K,
    )
    expert = (
        p_sorted_expert_ids[2 * e_idx + 1]
        if const_expr(_task_table)
        else fxh.view_as_torch_tensor(p_sorted_expert_ids, (1,), fx.Int32)[e_idx]
    )
    weights = fx.rocdl.make_buffer_tensor(
        fx.make_view(
            fx.recast_iter(fx.Float8E4M3FNUZ, fxh._as_ptr(p_weight))
            + fx.Int64(expert) * N * K,
            fx.make_layout(N * K, 1),
        ),
        max_size=False,
        num_records_bytes=N * K,
    )
    mm = ops.create_thr_mma(fx.Float8E4M3FNUZ, (1, 8, 1))
    atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))
    c = mm.make_fragment_C(
        fx.make_view(fx.get_iter(a), fx.make_ordered_layout((BN, BM), (0, 1)))
    )
    routing_scale = fxh.view_as_torch_tensor(
        fxh._as_ptr(p_sorted_weights)
        + (fx.Int64(row_begin) if const_expr(_task_table) else fx.Int64(e_idx) * BM),
        (BM,),
        fx.Float32,
    )
    row_tensor = fx.make_view(
        fx.get_iter(routing_scale), fx.make_layout((BN, BM), (0, 1))
    )
    row_scale = ops.load_tiled_mma_fragC(mm, row_tensor, copy_atom_bits=32)
    if const_expr(act_quant_type == "ptpc"):
        coords = ops.load_tiled_mma_fragC(
            mm,
            fx.make_view(fx.get_iter(ids_lds), fx.make_layout((BN, BM), (0, 1))),
            copy_atom_bits=32,
        )
        als = fx.rocdl.make_buffer_tensor(
            fxh.view_as_torch_tensor(p_a_scale, (M, TOPK), fx.Float32),
            max_size=False,
            num_records_bytes=fx.Int64(M) * TOPK * 4,
        )
        scale_copy = ops.get_buffer_copy_atom(fx.Float32, 32)
        ascale = mm.make_fragment_C(row_tensor)
        for dst, coord in fxh.all_elements(
            ops.get_tiled_mma_retile(mm, ascale, "C", copy_atom=scale_copy), coords
        ):
            encoded = coord[0].bitcast(fx.Uint32)
            fx.copy(
                scale_copy,
                fxh.atom_tensor(als, (encoded & 0xFFFFFF, encoded >> 24), 32),
                dst,
            )
        row_scale.store(row_scale.load() * ascale.load())
        if const_expr(weight_quant_type == "per_tensor"):
            ws = fx.make_view(
                fxh._as_ptr(p_w_scale, fx.Float32) + expert, fx.make_layout(1, 1)
            )[0]
            row_scale.store(row_scale.load() * ws)
    else:
        scalar_a = fx.make_view(
            fxh._as_ptr(p_a_scale, fx.Float32), fx.make_layout(1, 1)
        )[0]
        scalar_w = fx.make_view(
            fxh._as_ptr(p_w_scale, fx.Float32) + expert, fx.make_layout(1, 1)
        )[0]
        row_scale.store(row_scale.load() * (scalar_a * scalar_w))

    read_b_packet_g2r = partial(read_b_g2r_packet, weights)
    # Closures bind views and resident A; SSA values changing across N still pass through explicit state.
    afragments = out = store_atom = None
    scratch_write = scratch_read = scratch_base = lane_group = scale_buffer = None

    def b_startup_coords(q):
        # Locate startup Q1/Q2 only; K320 has four steps per N, so no validity field is needed.
        target = q + 1
        n, ks = target // (2 * KS), target % KS
        return n, ks, (target % (2 * KS)) // KS, (n * KS + ks) & 1

    def b_offsets(ks):
        width = K_WIDTHS[ks]
        index = group * (BN * width // 64) + group_tid
        global_offset = (
            (index // width) * (16 * K)
            + ((index % width) // 16) * 256
            + (index % 16) * 16
        )
        local_offset = (
            (index // width) * (16 * width)
            + ((index % width) // 16) * 256
            + (index % 16) * 16
        )
        return global_offset, local_offset

    def tail192_b_offsets(part):
        # Map contiguous LDS bytes back to full-K320 preshuffle; take only K[128:320] per 16 rows.
        local_offset = tid * 16 if part == 0 else 8192 + tid * 8
        global_offset = (local_offset // (16 * 192)) * (16 * K) + local_offset % (
            16 * 192
        )
        return global_offset, local_offset

    # B callbacks use the shared tile signature; kb selects fixed LDS slots, with no extra P-slot adapter.
    def read_b_g2r(n, kb, half, *, prefetch_slot=0):
        if const_expr(K_WIDTHS[kb] == 192):
            scalar = n * BN * K + K_OFFSETS[kb] * 16 + half * (BN // 2) * K
            pieces = [
                read_b_packet_g2r(
                    tail192_b_offsets(part)[0], scalar, 4 if part == 0 else 2
                )
                for part in range_constexpr(2)
            ]
            # Carry six actual u32 values together for the common N loop; joining adds no K loads/MFMA.
            fragment = fx.make_rmem_tensor(fx.make_layout(6, 1), fx.Uint32)
            fragment.store(
                Vec.from_elements(
                    [Vec(pieces[0].load())[i] for i in range_constexpr(4)]
                    + [Vec(pieces[1].load())[i] for i in range_constexpr(2)],
                    fx.Uint32,
                )
            )
            return fragment
        # Each group has 256 threads loading 16B each for BK128; no conditional exec or redundant full loads.
        return read_b_packet_g2r(
            b_offsets(kb)[0],
            n * BN * K + K_OFFSETS[kb] * 16 + half * (BN // 2) * K,
            K_WIDTHS[kb] // 32,
        )

    def store_b_r2s(lds_slot, kb, half, *, prefetch_slot=0, fragment, address=None):
        assert address is None
        width = K_WIDTHS[kb]
        base = fx.recast_iter(fx.Uint32, bptrs[kb]) + half * (BN // 2) * (width // 4)

        if const_expr(width == 192):
            values = Vec(fragment.load())
            for part in range_constexpr(2):
                words, start = (4, 0) if part == 0 else (2, 4)
                piece = fx.make_rmem_tensor(fx.make_layout(words, 1), fx.Uint32)
                piece.store(values.shuffle(values, list(range(start, start + words))))
                destination = fx.make_view(
                    base + tail192_b_offsets(part)[1] // 4, fx.make_layout(words, 1)
                )
                fx.copy(
                    ops.get_universal_copy_atom(fx.Uint32, words * 32),
                    piece,
                    destination,
                )
        else:
            destination = fx.make_view(
                base + b_offsets(kb)[1] // 4, fx.make_layout(4, 1)
            )
            fx.copy(ops.get_universal_copy_atom(fx.Uint32, 128), fragment, destination)

    def read_b_s2r(lds_slot, kb, half, *, packet, address=None):
        assert address is None
        width = K_WIDTHS[kb]
        # Fixed slot widths 128/192 give half strides of 8192/12288B, respectively.
        view = fx.make_view(
            bptrs[kb] + half * (BN // 2) * width + packet * (BN // 4) * width,
            fx.make_layout(
                ((16, BN // 64), (16, width // 16)), ((16, 16 * width), (1, 256))
            ),
        )

        return ops.load_tiled_mma_fragA(mm, view, copy_atom_bits=128)

    def prepare_views():
        nonlocal out, store_atom, scratch_write, scratch_read, scratch_base, lane_group, scale_buffer
        out = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fxh._as_ptr(p_output, fx.BFloat16)
                + (
                    fx.Int64(row_begin) * STRIDE
                    if const_expr(_task_table)
                    else fx.Int64(e_idx) * BM * STRIDE
                ),
                fx.make_layout((N, BM), (1, STRIDE)),
            ),
            max_size=False,
            num_records_bytes=BM * STRIDE * 2,
        )
        store_atom = fx.make_copy_atom(
            fx.rocdl.BufferCopy128b(cache_modifier=_store_cache), fx.BFloat16
        )
        scratch_write = ops.get_universal_copy_atom(fx.BFloat16, 128)
        scratch_read = ops.get_universal_copy_atom(fx.BFloat16, 64)
        scratch_base = (wave % 4) * 16 * BN
        lane_group = lane // 16
        if const_expr(weight_quant_type == "ptpc"):
            scale_buffer = fx.rocdl.make_buffer_tensor(
                fxh.view_as_torch_tensor(
                    fxh._as_ptr(p_w_scale, fx.Float32) + fx.Int64(expert) * N,
                    (N,),
                    fx.Float32,
                ),
                max_size=False,
                num_records_bytes=N * 4,
            )

    def read_scale_g2r(n, pair):
        return read_8x1_scale_g2r(
            n,
            pair,
            weight_quant_type=weight_quant_type,
            scale_buffer=scale_buffer,
            mm=mm,
            ops=ops,
        )

    # K320 consumes packets only; the shared full-half branch is unreachable, so no full_half parameter.
    def mma(weight, kb, record):
        return mfma_8x1_record(
            weight, kb, record, k_widths=K_WIDTHS, afragments=afragments, c=c, atom=atom
        )

    def shuffle_c_r2s2r(n, packed, row, half):
        return shuffle_8x1_c_r2s2r(
            n,
            packed,
            row,
            half,
            scratch_view=scratch_view,
            scratch_base=scratch_base,
            lane=lane,
            lane_group=lane_group,
            wave=wave,
            out=out,
            scratch_write=scratch_write,
            scratch_read=scratch_read,
        )

    def store_c_r2g(fragments, destinations, lgkmcnt=0):
        return store_8x1_c_r2g(
            fragments, destinations, lgkmcnt=lgkmcnt, store_atom=store_atom
        )

    # Fixed compile-time bindings; no GPU operations are generated.
    pack = partial(
        pack_8x1_record, c=c, row_scale=row_scale, weight_quant_type=weight_quant_type
    )
    store_c_tile_r2g = partial(
        store_8x1_c_tile_r2g, shuffle_c_r2s2r=shuffle_c_r2s2r, store_c_r2g=store_c_r2g
    )
    clear = partial(clear_8x1_record, c=c)
    schedule_pack = partial(schedule_k128_pack, weight_quant_type=weight_quant_type)
    save_state = partial(
        save_8x1_state,
        c=c,
        ptpc=weight_quant_type == "ptpc",
        first_unpacked=FIRST_UNPACKED,
        prepare_b_addresses=None,
    )
    run_tile = partial(
        run_8x1_tile,
        k=320,
        n_tiles=NT,
        ptpc=weight_quant_type == "ptpc",
        read_b_g2r=read_b_g2r,
        store_b_r2s=store_b_r2s,
        read_b_s2r=read_b_s2r,
        read_scale_g2r=read_scale_g2r,
        pack=pack,
        shuffle_c_r2s2r=shuffle_c_r2s2r,
        store_c_r2g=store_c_r2g,
        mma=mma,
        clear=clear,
        schedule_pack=schedule_pack,
        k_widths=K_WIDTHS,
        prepare_b_addresses=None,
    )

    def prologue():
        nonlocal afragments
        prepare_views()
        acopy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float8E4M3FNUZ)
        afragments = [
            mm.make_fragment_B(
                fx.make_view(
                    fx.get_iter(a),
                    fx.make_layout((BM, width), (1, BM)),
                )
            )
            for width in K_WIDTHS
        ]
        first_index = group * 256 + group_tid
        first_offset = (
            (first_index // K_WIDTHS[0]) * (16 * K)
            + ((first_index % K_WIDTHS[0]) // 16) * 256
            + (first_index % 16) * 16
        )
        prefetch0_n, prefetch0_kb, prefetch0_half, _prefetch0_lds_slot = (
            b_startup_coords(0)
        )
        prefetch1_n, prefetch1_kb, prefetch1_half, _prefetch1_lds_slot = (
            b_startup_coords(1)
        )

        # Execute: first B -> A -> Q0 to LDS -> Q1/Q2; A loads 128+192, with no prologue MMA.
        first_b = read_b_packet_g2r(first_offset, 0, 4)
        rocdl.sched_barrier(0)
        read_a_g2r(a, ids_lds, afragments, acopy, lane, wave, K_WIDTHS, K_OFFSETS)
        rocdl.s_waitcnt(vmcnt=4)
        store_b_r2s(0, 0, 0, prefetch_slot=0, fragment=first_b)
        rocdl.sched_barrier(0)
        b_prefetch = [
            read_b_g2r(prefetch0_n, prefetch0_kb, prefetch0_half, prefetch_slot=0),
            None,
        ]
        b_prefetch[1] = read_b_g2r(
            prefetch1_n, prefetch1_kb, prefetch1_half, prefetch_slot=1
        )
        rocdl.s_waitcnt(vmcnt=1)
        # Stagger before q0; Q0 writes from both groups must first be ready in LDS.
        rocdl.s_waitcnt(lgkmcnt=0)
        stage_end()
        c.fill(0)
        return b_prefetch

    def prepare_loop_state(b_prefetch, packed, scales, b_addresses):
        # Carriers still start after N1 completes; preserve the initial save order.
        b_carriers = [
            fx.make_fragment_like(b_prefetch[index]) for index in range_constexpr(2)
        ]
        scale_carriers = None
        if const_expr(weight_quant_type == "ptpc"):
            scale_carriers = [
                fx.make_fragment_like(scales[pair])
                for pair in range_constexpr(FIRST_UNPACKED, 4)
            ]
        restore_state = partial(
            restore_8x1_state,
            c=c,
            ptpc=weight_quant_type == "ptpc",
            first_unpacked=FIRST_UNPACKED,
            prepare_b_addresses=None,
            b_carriers=b_carriers,
            scale_carriers=scale_carriers,
        )

        return save_state(b_prefetch, packed, scales, b_addresses), restore_state

    # Pipeline: directly reuse run_8x1_tile following the K128n outer body, without Memory/Compute wrappers.
    # Prologue keeps A resident, LDS=Q0, P0/P1=Q1/Q2.
    b_prefetch = prologue()

    # N0: stagger before q0; pack C0/C1, leaving FP32 C2/C3 for the next N.
    if group == 1:
        stage_end()
    b_prefetch, c_bf16, c_scales, b_addresses = run_tile(
        0,
        b_prefetch,
        previous_packed=[],
        previous_scales=[],
        first=True,
        last=NT == 1,
        addresses=[],
    )

    # N1: finish packing/drain N0 while producing N1; retain separate budgets and old scale tails.
    if const_expr(NT > 1):
        b_prefetch, c_bf16, c_scales, b_addresses = run_tile(
            1,
            b_prefetch,
            c_bf16,
            c_scales[FIRST_UNPACKED:],
            first=False,
            last=NT == 2,
            addresses=b_addresses,
        )

    # Loop: retire old C/compute new C with a fixed one-N backedge; NT=3 stays zero-trip.
    if const_expr(NT >= 3):
        initial, restore_state = prepare_loop_state(
            b_prefetch, c_bf16, c_scales, b_addresses
        )
        for block_start, state in range(2, LOOP_END, 1, init=initial):
            b_prefetch, previous_packed, previous_scales, b_addresses = restore_state(
                state
            )
            b_prefetch, packed, scales, b_addresses = run_tile(
                fx.Int64(block_start) + 0,
                b_prefetch,
                previous_packed,
                previous_scales,
                first=False,
                last=False,
                addresses=b_addresses,
            )
            results = yield save_state(b_prefetch, packed, scales, b_addresses)
        b_prefetch, c_bf16, previous_scales, b_addresses = restore_state(results)
    else:
        previous_scales = c_scales[FIRST_UNPACKED:]

    # Tail: still interleave old/new C; prune only B without consumers. Empty for NT=1/2.
    for n in range_constexpr(LOOP_END, NT):
        b_prefetch, c_bf16, c_scales, b_addresses = run_tile(
            n,
            b_prefetch,
            c_bf16,
            previous_scales,
            first=False,
            last=n == NT - 1,
            addresses=b_addresses,
        )
        previous_scales = c_scales[FIRST_UNPACKED:]

    # Epilogue: wait, finish packing C2/C3 with the final tile's full scales, then drain C stores.
    rocdl.s_waitcnt(vmcnt=0)
    for pair in range_constexpr(FIRST_UNPACKED, 4):
        c_bf16.append(pack(pair, c_scales[pair]))
    store_c_tile_r2g(NT - 1, c_bf16)
    stage_end()
    if group == 0:
        stage_end()
