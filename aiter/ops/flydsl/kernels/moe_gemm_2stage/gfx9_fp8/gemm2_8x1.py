# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""8x1 Down kernels: shared primitives, task setup and N schedule, the entry point, then K128n and K192."""

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
from .common import (
    BufferTensor,
    FlyObjCache,
    LdsTensor,
    _f32_to_bf16,
    eltwise_op,
    get_down_device_config,
)


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


def _pin_vgpr(value):
    # Empty tied asm emits no ISA but makes the value ready before the next sched barrier.
    return fx.Int32(
        llvm.inline_asm(
            ir.IntegerType.get_signless(32),
            [as_ir_value(value)],
            "",
            "=v,0",
            has_side_effects=True,
        )
    )


# ==================== Shared task setup ====================
def _input_tensor(p_input, M, TOPK, K):
    return fx.rocdl.make_buffer_tensor(
        fxh.view_as_torch_tensor(p_input, (M, TOPK, K), fx.Float8E4M3FNUZ),
        max_size=False,
        num_records_bytes=fx.Int64(M) * TOPK * K,
    )


def _sorted_ids_tensor(p_sorted_ids, row_begin, rows):
    return fx.rocdl.make_buffer_tensor(
        fxh.view_as_torch_tensor(
            fxh._as_ptr(p_sorted_ids) + fx.Int64(row_begin), (rows,), fx.Int32
        ),
        max_size=False,
        num_records_bytes=rows * 4,
    )


def _routing_scale(p_sorted_weights, row_begin, rows):
    return fxh.view_as_torch_tensor(
        fxh._as_ptr(p_sorted_weights) + fx.Int64(row_begin), (rows,), fx.Float32
    )


def _expert_weights(p_weight, expert, N, K):
    return fx.rocdl.make_buffer_tensor(
        fx.make_view(
            fx.recast_iter(fx.Float8E4M3FNUZ, fxh._as_ptr(p_weight))
            + fx.Int64(expert) * N * K,
            fx.make_layout(N * K, 1),
        ),
        max_size=False,
        num_records_bytes=N * K,
    )


def _load_row_scale(
    ops,
    mm,
    routing_scale,
    sorted_ids_lds,
    p_a_scale,
    p_w_scale,
    expert,
    M,
    TOPK,
    *,
    block_n,
    block_m,
    weight_quant_type,
    act_quant_type,
):
    """Load routing weights in the C layout and fold in the activation and weight scales."""
    row_tensor = fx.make_view(
        fx.get_iter(routing_scale), fx.make_layout((block_n, block_m), (0, 1))
    )
    row_scale = ops.load_tiled_mma_fragC(mm, row_tensor, copy_atom_bits=32)
    if act_quant_type == "ptpc":
        coords = ops.load_tiled_mma_fragC(
            mm,
            fx.make_view(
                fx.get_iter(sorted_ids_lds), fx.make_layout((block_n, block_m), (0, 1))
            ),
            copy_atom_bits=32,
        )
        a_scale = fx.rocdl.make_buffer_tensor(
            fxh.view_as_torch_tensor(p_a_scale, (M, TOPK), fx.Float32),
            max_size=False,
            num_records_bytes=fx.Int64(M) * TOPK * 4,
        )
        copy_atom = ops.get_buffer_copy_atom(fx.Float32, 32)
        token_scale = mm.make_fragment_C(row_tensor)
        for dst, coord in fxh.all_elements(
            ops.get_tiled_mma_retile(mm, token_scale, "C", copy_atom=copy_atom), coords
        ):
            encoded = coord[0].bitcast(fx.Uint32)
            fx.copy(
                copy_atom,
                fxh.atom_tensor(a_scale, (encoded & 0xFFFFFF, encoded >> 24), 32),
                dst,
            )
        row_scale.store(row_scale.load() * token_scale.load())
        if weight_quant_type == "per_tensor":
            weight_scale = fx.make_view(
                fxh._as_ptr(p_w_scale, fx.Float32) + expert, fx.make_layout(1, 1)
            )[0]
            row_scale.store(row_scale.load() * weight_scale)
    else:
        # Fold the global activation scale and the per-expert weight scale in advance.
        act_scale = fx.make_view(
            fxh._as_ptr(p_a_scale, fx.Float32), fx.make_layout(1, 1)
        )[0]
        weight_scale = fx.make_view(
            fxh._as_ptr(p_w_scale, fx.Float32) + expert, fx.make_layout(1, 1)
        )[0]
        row_scale.store(row_scale.load() * (act_scale * weight_scale))
    return row_scale


def _output_tensor(p_output, row_begin, N, rows, stride):
    return fx.rocdl.make_buffer_tensor(
        fx.make_view(
            fxh._as_ptr(p_output, fx.BFloat16) + fx.Int64(row_begin) * stride,
            fx.make_layout((N, rows), (1, stride)),
        ),
        max_size=False,
        num_records_bytes=rows * stride * 2,
    )


def _weight_scale_buffer(p_w_scale, expert, N):
    return fx.rocdl.make_buffer_tensor(
        fxh.view_as_torch_tensor(
            fxh._as_ptr(p_w_scale, fx.Float32) + fx.Int64(expert) * N, (N,), fx.Float32
        ),
        max_size=False,
        num_records_bytes=N * 4,
    )


def _make_a_fragments(mm, a, rows, k_widths):
    """Return the A copy atom and one resident A fragment per K block."""
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float8E4M3FNUZ)
    fragments = [
        mm.make_fragment_B(
            fx.make_view(fx.get_iter(a), fx.make_layout((rows, width), (1, rows)))
        )
        for width in k_widths
    ]
    return copy_atom, fragments


def _b_coords(n, target, ks):
    """(n, kb, half, lds_slot) of B request Q[target], counted from N tile n."""
    target_n, target_kb = n + target // (2 * ks), target % ks
    return (
        target_n,
        target_kb,
        (target % (2 * ks)) // ks,
        (target_n * ks + target_kb) & 1,
    )


# ==================== Shared 8x1 event accounting and SSA ====================
def packing_events(k, stage, first=False):
    """Return (packet, previous-N, super-record), not the current MFMA output fragment."""
    assert k in (256, 384, 512, 640), "shared 8x1 schedule supports only K128n"
    ks = k // 128
    if stage == 0 and not first:
        return ((0, True, 3),)
    if stage == ks - 1:
        return ((1, False, 0),)
    if stage == ks:
        return ((0, False, 1),)
    if stage == 2 * ks - 1:
        return ((1, False, 2),)
    return ()


@cache
def vmem_wait_schedule(k, n_tiles, ptpc=True):
    """Per-step vmcnt budgets, counting one event per buffer/global load.

    A budget is the minimum age of the B and scale loads consumed next; scales
    count because some K256 packets use the previous step's scale.
    """
    assert (
        k in (256, 384, 512, 640) and n_tiles >= 1
    ), "shared 8x1 schedule does not support this K or N tile count"
    ks, events, requests, scales = k // 128, [], {}, {}

    def valid(q):
        # Request q submits Q[q+1]; prune only requests beyond the final consumer.
        return 0 <= q < n_tiles * 2 * ks - 1

    def request(q):
        if valid(q):
            events.append(("B", q))
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
        store_count = 2 if n > 0 and stage < 4 else 0
        events.extend([("store", n, stage)] * store_count)
        required = [requests[q]] if valid(q) else []
        if ptpc:
            for _, previous, record in packing_events(k, stage, n == 0):
                required.append(scales[n - int(previous), record])
        budget = min((len(events) - 1 - index for index in required), default=63)
        result.append(budget)
        request(q + 2)
    return tuple(result)


def _b_parts(b_prefetch):
    # K192 carries each B half as 16B+8B parts; other K families carry one fragment per half.
    return [
        part
        for carrier in b_prefetch
        for part in (carrier if isinstance(carrier, list) else [carrier])
    ]


def save_8x1_state(b_prefetch, packed, scales, addresses, *, c, ptpc, first_unpacked):
    # Keep FP8 payloads as integer bits across SCF loop boundaries.
    state = [part.load().bitcast(fx.Uint32) for part in _b_parts(b_prefetch)]
    for row in range_constexpr(2):
        for group in range_constexpr(2 * first_unpacked, 8):
            state.append(c[None, group, row].load())
    if const_expr(ptpc):
        for pair in range_constexpr(first_unpacked, 4):
            state.append(scales[pair].load())
    for pair in range_constexpr(first_unpacked):
        for row in range_constexpr(2):
            state.append(packed[pair][row])
    state.extend(addresses)
    return state


def restore_8x1_state(state, *, c, ptpc, first_unpacked, b_carriers, scale_carriers):
    carriers = _b_parts(b_carriers)
    for carrier, value in zip(carriers, state):
        carrier.store(value.bitcast(carrier.dtype))
    offset = len(carriers)
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
    # Carried B addresses (K192 and K128n KS=3/5 only) are the state tail.
    addresses = [fx.Int32(value) for value in state[offset:]]
    b_prefetch = [
        list(carrier) if isinstance(carrier, list) else carrier
        for carrier in b_carriers
    ]
    return b_prefetch, packed, scales, addresses


def _prepare_8x1_loop_state(
    b_prefetch, packed, scales, b_addresses, *, c, ptpc, first_unpacked, save_state
):
    # Create carriers only after N1 completes; do not extend cross-N SSA lifetimes earlier.
    b_carriers = [
        (
            [fx.make_fragment_like(part) for part in carrier]
            if isinstance(carrier, list)
            else fx.make_fragment_like(carrier)
        )
        for carrier in b_prefetch
    ]
    scale_carriers = None
    if const_expr(ptpc):
        scale_carriers = [
            fx.make_fragment_like(scales[pair])
            for pair in range_constexpr(first_unpacked, 4)
        ]
    restore_state = partial(
        restore_8x1_state,
        c=c,
        ptpc=ptpc,
        first_unpacked=first_unpacked,
        b_carriers=b_carriers,
        scale_carriers=scale_carriers,
    )
    return save_state(b_prefetch, packed, scales, b_addresses), restore_state


# ==================== K128n N tile and shared N schedule ====================
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
    prepare_b_addresses=None,
):
    """Run one N tile; q = 2*KS*n + step orders events and packing_events drives packing.

    Completes previous_packed in place and returns the current N's packed records and
    all four scales; previous_scales holds only the tail, indexed by record - FIRST_UNPACKED.
    """
    KS = k // 128
    budgets = vmem_wait_schedule(k, n_tiles, ptpc)
    FIRST_UNPACKED = 3

    def has_b_target(step, *, ahead, last):
        return not last or step + ahead < 2 * KS

    packed, scales = [], []
    for step in range_constexpr(2 * KS):
        # kb indexes BK128 blocks; it is not an offset or slot.
        # half=0/1 selects the first/last 64 columns of N128; visit all L K blocks before H.
        kb, half, prefetch_slot = step % KS, step // KS, step & 1
        lds_slot = (n * KS + kb) & 1
        # N0 q0 reads a whole N64 half; all other steps read two N32 packets.
        read_full_half = first and step == 0
        has_output = not first and step < 4
        events = packing_events(k, step, first=first)

        # Memory: read Q[q], interleaving previous-N C stores; P and LDS slots rotate independently.
        priority(0)
        if const_expr(step < 4):
            scales.append(read_scale_g2r(n, step))
        if const_expr(has_output):
            # Step s stores previous-N output quarter (row=s%2, half=s//2), not an N32 record.
            fragments, destinations = shuffle_c_r2s2r(
                n - 1, previous_packed, step % 2, step // 2
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
                store_c_r2g(fragments, destinations, lgkmcnt=4)
                fx.rocdl.sched_barrier(0)
            b1 = read_b_s2r(lds_slot, kb, half, packet=1, address=b_read_address)

        # Protect the next B submission and this step's pack scales; N1 is separate, the backedge uses n>=2.
        budget_n = (
            0 if first else n_tiles - 1 if last else n if isinstance(n, int) else 2
        )
        if const_expr(has_b_target(step, ahead=1, last=last) or (ptpc and events)):
            rocdl.s_waitcnt(vmcnt=budgets[budget_n * 2 * KS + step])
        if const_expr(has_b_target(step, ahead=1, last=last)):
            # B r->s submits Q[q+1]; materialize the target after waiting, consuming P[prefetch_slot].
            _b_r2s_n, b_r2s_kb, b_r2s_half, b_r2s_lds_slot = _b_coords(n, step + 1, KS)
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
            # Prefetch Q[q+3] into the submitted P slot; materialize its target before the sched barrier.
            b_g2r_n, b_g2r_kb, b_g2r_half, _b_g2r_lds_slot = _b_coords(n, step + 3, KS)
            fx.rocdl.sched_barrier(0)
            b_prefetch[prefetch_slot] = read_b_g2r(
                b_g2r_n, b_g2r_kb, b_g2r_half, prefetch_slot=prefetch_slot
            )
        if const_expr(first and step == 0):
            # Q1's cross-group consumers need both LDS writes.
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


@flyc.jit
def _run_8x1_pipeline(
    b_prefetch,
    b_addresses,
    group,
    n_tiles,
    tail_tiles,
    first_unpacked,
    run_tile,
    prepare_loop_state,
    save_state,
    pack,
    store_c_tile_r2g,
):
    """Run N0, N1, the dynamic N loop and the last tail_tiles N tiles, then drain C.

    Records below first_unpacked are packed within their N; the rest stay FP32 until the next N.
    """
    loop_end = n_tiles - tail_tiles
    # N0: stagger the groups before q0; group1's wait pairs with group0's final barrier.
    if group == 1:
        stage_end()
    b_prefetch, c_bf16, c_scales, b_addresses = run_tile(
        0, b_prefetch, [], [], first=True, addresses=b_addresses
    )

    # N1: finish packing/drain N0 while producing N1; pass only the unpacked records' old scales.
    b_prefetch, c_bf16, c_scales, b_addresses = run_tile(
        1, b_prefetch, c_bf16, c_scales[first_unpacked:], addresses=b_addresses
    )

    # Loop: retire old C/compute new C with a one-N backedge; a zero-trip loop keeps its state.
    initial, restore_state = prepare_loop_state(
        b_prefetch, c_bf16, c_scales, b_addresses
    )
    for block_start, state in range(2, loop_end, 1, init=initial):
        b_prefetch, previous_packed, previous_scales, b_addresses = restore_state(state)
        b_prefetch, packed, scales, b_addresses = run_tile(
            fx.Int64(block_start) + 0,
            b_prefetch,
            previous_packed,
            previous_scales,
            addresses=b_addresses,
        )
        results = yield save_state(b_prefetch, packed, scales, b_addresses)
    b_prefetch, c_bf16, previous_scales, b_addresses = restore_state(results)

    # Tail: still interleave old/new C; prune only B without consumers.
    for n in range_constexpr(loop_end, n_tiles):
        b_prefetch, c_bf16, c_scales, b_addresses = run_tile(
            n,
            b_prefetch,
            c_bf16,
            previous_scales,
            last=n == n_tiles - 1,
            addresses=b_addresses,
        )
        previous_scales = c_scales[first_unpacked:]

    # Epilogue: wait, pack the remaining records, store all four output quarters, and keep
    # group0's compensating barrier.
    rocdl.s_waitcnt(vmcnt=0)
    for pair in range_constexpr(first_unpacked, 4):
        c_bf16.append(pack(pair, c_scales[pair]))
    store_c_tile_r2g(n_tiles - 1, c_bf16)
    stage_end()
    if group == 0:
        stage_end()


# ==================== Entry point ====================
def _build_moe_gemm2_8x1(
    *, N, K, weight_quant_type, act_quant_type, TOPK, tile_k, padding, store_cache
):
    """M256 full-task kernel of the compact Down path; the compact builder validates shapes."""
    assert tile_k in (None, 128) or (
        K == 192 and tile_k == 192
    ), "8x1 keeps one K192 block of 192 and BK128 for other K values"
    # K192 uses one block of 192; accept callers passing tile_k=128.
    tile_k = 192 if K == 192 else 128
    assert store_cache in (0, 2), "store aux: 0=normal, 2=SLC"

    BM = 256
    ops = FlyObjCache(cache_ir=False)
    topology, xcc_count = get_down_device_config()
    se_count = xcc_count * 4
    emit_body = _emit_k192_body if K == 192 else _emit_k128n_body

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
            emit_body(
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
                padding,
                weight_quant_type,
                act_quant_type,
                store_cache,
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
    padding,
    weight_quant_type,
    act_quant_type,
    _store_cache,
    ops,
):
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
    WEIGHT_QUARTER_ATOMS = BN * BK // (16 * 4)
    WEIGHT_PREFETCH_SLOTS = 2
    OUTPUT_STORES_PER_WAVE = WAVE_M * BN * 2 // (64 * 16)
    CSHUFFLE_N_PAIRS = BN // 32
    STRIDE = N + padding // 2
    CSHUFFLE_WAVES = 4

    assert WEIGHT_QUARTER_ATOMS == 256
    assert OUTPUT_STORES_PER_WAVE == 8

    row_begin = p_sorted_expert_ids[2 * e_idx]
    input_tensor = _input_tensor(p_input, M, TOPK, K)
    sorted_ids = _sorted_ids_tensor(p_sorted_ids, row_begin, BM)
    sorted_weights = _routing_scale(p_sorted_weights, row_begin, BM)
    expert_id = p_sorted_expert_ids[2 * e_idx + 1]

    shared_allocator = fx.SharedAllocator()
    weight_slots = shared_allocator.allocate(
        fx.Array[fx.Float8E4M3FNUZ, 2 * BN * BK, 16]
    )
    weight_storage_ptrs = [weight_slots.peek().ptr, weight_slots.peek().ptr + BN * BK]
    cshuffle_storage = shared_allocator.allocate(
        fx.Array[fx.BFloat16, CSHUFFLE_WAVES * 16 * BN, 16]
    )
    sorted_lds = fx.make_view(
        fx.recast_iter(fx.Int32, cshuffle_storage.peek().ptr), fx.make_layout(BM, 1)
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
    frag_row_scale = _load_row_scale(
        ops,
        mm,
        sorted_weights,
        sorted_lds,
        p_a_scale,
        p_w_scale,
        expert_id,
        M,
        TOPK,
        block_n=BN,
        block_m=BM,
        weight_quant_type=weight_quant_type,
        act_quant_type=act_quant_type,
    )

    weight_flat = _expert_weights(p_weight, expert_id, N, K)
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

    # Whole-half views serve only the N0 q0 read.
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
            lds_offset = (
                half * (BN // 2) * BK
                + quarter_n_group * (16 * BK)
                + quarter_k_group * 256
                + quarter_n_inner * 16
            )
            destination = fx.make_view(
                weight_storage_ptrs[0] + lds_slot * BN * BK + lds_offset,
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
        # Write both records to their LDS planes, then read two 8B pieces per output row.
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
        return [_pin_vgpr(value) for value in values]

    def read_b_s2r(lds_slot, kb, half, *, packet, address=None):
        if const_expr(packet is None):
            # N0 q0 reads a whole half with its own register layout.
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
        view = weight_lds_quarter_view(
            weight_storage_ptrs[0] + lds_slot * BN * BK, 2 * half + packet
        )
        return ops.load_tiled_mma_fragA(mm, view, copy_atom_bits=128)

    # KS=3/5 flips LDS slots across N; carry full addresses. Even KS needs no callback.
    prepare_addresses = prepare_b_addresses if const_expr(K in (384, 640)) else None
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

    output_tensor = _output_tensor(p_output, row_begin, N, BM, STRIDE)
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
                    output_row_half * 8 + lane // 8, (output_atom % 2) * 2, n_group // 2
                )
                + (n_group % 2) * 4
            )
            cshuffle_read_bases.append(fx.get_iter(cshuffle_lds) + offset)

    weight_scale_buffer = None
    if const_expr(weight_quant_type == "ptpc"):
        weight_scale_buffer = _weight_scale_buffer(p_w_scale, expert_id, N)
    input_copy, a_fragments = _make_a_fragments(mm, input_tensor, BM, K_WIDTHS)
    cshuffle_read_pointers = []

    def pin_c_read_addresses():
        for index in range_constexpr(4):
            pointer = cshuffle_read_bases[index]
            address = _pin_vgpr(fx.Int32(fx.ptrtoint(pointer)))
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

    def schedule_pack():
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

    pack = partial(
        pack_8x1_record,
        c=frag_c,
        row_scale=frag_row_scale,
        weight_quant_type=weight_quant_type,
    )
    store_c_r2g = partial(store_8x1_c_r2g, store_atom=output_store_atom)
    store_c_tile_r2g = partial(
        store_8x1_c_tile_r2g, shuffle_c_r2s2r=shuffle_c_r2s2r, store_c_r2g=store_c_r2g
    )
    save_state = partial(
        save_8x1_state,
        c=frag_c,
        ptpc=weight_quant_type == "ptpc",
        first_unpacked=FIRST_UNPACKED,
    )
    run_tile = partial(
        run_8x1_tile,
        k=K,
        n_tiles=NT,
        ptpc=weight_quant_type == "ptpc",
        read_b_g2r=read_b_g2r,
        store_b_r2s=store_b_r2s,
        read_b_s2r=read_b_s2r,
        read_scale_g2r=partial(
            read_8x1_scale_g2r,
            weight_quant_type=weight_quant_type,
            scale_buffer=weight_scale_buffer,
            mm=mm,
            ops=ops,
        ),
        pack=pack,
        shuffle_c_r2s2r=shuffle_c_r2s2r,
        store_c_r2g=store_c_r2g,
        mma=mma,
        clear=partial(clear_8x1_record, c=frag_c),
        schedule_pack=schedule_pack,
        prepare_b_addresses=prepare_addresses,
    )

    def prologue():
        # B (n,kb,half) denotes N128 tile, K block, and N64 half (0=L/1=H); slot is the buffer slot.
        # Startup prefetches Q1/Q2; at least four steps per N guarantee their consumers.
        prefetch0_n, prefetch0_kb, prefetch0_half, _ = _b_coords(0, 1, KS)
        prefetch1_n, prefetch1_kb, prefetch1_half, _ = _b_coords(0, 2, KS)

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

    prepare_loop_state = partial(
        _prepare_8x1_loop_state,
        c=frag_c,
        ptpc=weight_quant_type == "ptpc",
        first_unpacked=FIRST_UNPACKED,
        save_state=save_state,
    )

    # Pipeline: prologue keeps A resident, LDS=Q0, P0/P1=Q1/Q2; N0 packs C0-C2, leaving FP32 C3.
    b_prefetch = prologue()
    b_addresses = (
        prepare_b_addresses(0) if const_expr(prepare_addresses is not None) else []
    )
    _run_8x1_pipeline(
        b_prefetch,
        b_addresses,
        group,
        NT,
        1,
        FIRST_UNPACKED,
        run_tile,
        prepare_loop_state,
        save_state,
        pack,
        store_c_tile_r2g,
    )


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
    group_tid,
    N,
    K,
    TOPK,
    tile_k,
    padding,
    weight_quant_type,
    act_quant_type,
    _store_cache,
    ops,
):
    BM, BN = 256, 128
    K_WIDTHS = (192,)
    K_OFFSETS = (0,)
    NT = N // BN
    STRIDE = N + padding // 2
    FIRST_UNPACKED = 2

    def schedule_pack():
        # PTPC has 40 VALU instructions, per-tensor 24; each K192 packet has 24 MFMA instructions.
        for index in range_constexpr(24):
            rocdl.sched_group_barrier(0x8, 1, 0)
            rocdl.sched_group_barrier(
                0x2, 2 if weight_quant_type == "ptpc" and index < 16 else 1, 0
            )
        rocdl.sched_barrier(0)

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
    ids = _sorted_ids_tensor(p_sorted_ids, row_begin, BM)
    if tid < BM:
        ids_lds[tid] = ids[tid]
    gpu.barrier()

    a = _input_tensor(p_input, M, TOPK, K)
    expert = p_sorted_expert_ids[2 * e_idx + 1]
    weights = _expert_weights(p_weight, expert, N, K)
    mm = ops.create_thr_mma(fx.Float8E4M3FNUZ, (1, 8, 1))
    atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))
    c = mm.make_fragment_C(
        fx.make_view(fx.get_iter(a), fx.make_ordered_layout((BN, BM), (0, 1)))
    )
    routing_scale = _routing_scale(p_sorted_weights, row_begin, BM)
    row_scale = _load_row_scale(
        ops,
        mm,
        routing_scale,
        ids_lds,
        p_a_scale,
        p_w_scale,
        expert,
        M,
        TOPK,
        block_n=BN,
        block_m=BM,
        weight_quant_type=weight_quant_type,
        act_quant_type=act_quant_type,
    )

    def read_b_packet_g2r(byte_offset, scalar_offset, words):
        source = BufferTensor(
            fx.make_view(fx.get_iter(weights), fx.make_layout(words * 4, 1))
        )
        fragment = fx.make_rmem_tensor(fx.make_layout(words, 1), fx.Uint32)
        values = source.load(
            voffset_bytes=fx.Int32(byte_offset), soffset_bytes=fx.Int32(scalar_offset)
        ).bitcast(fx.Uint32)
        fragment.store(values)
        return fragment

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
        return [_pin_vgpr(value) for value in values]

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

    # Build the partition once; full lane addresses are prepared at the previous compute tail and carried.
    b_template = fx.make_view(
        bptr, fx.make_layout(((16, BN // 64), (16, K // 16)), ((16, 16 * K), (1, 256)))
    )
    b_copy = ops.get_universal_copy_atom(fx.Float8E4M3FNUZ, 128)
    b_partition = ops.get_tiled_mma_partition_S(mm, b_template, "A", copy_atom_bits=128)
    b_write_ptr = fx.recast_iter(fx.Uint32, bptr)
    # The Uint32 templates preserve the pointer types/alignments of b_write_ptr and b_write_ptr+2.
    b_write_templates = [
        fx.make_view(b_write_ptr, fx.make_layout(4, 1)),
        fx.make_view(b_write_ptr + 2, fx.make_layout(2, 1)),
    ]
    out = _output_tensor(p_output, row_begin, N, BM, STRIDE)
    store_atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy128b(cache_modifier=_store_cache), fx.BFloat16
    )
    scratch_write = ops.get_universal_copy_atom(fx.BFloat16, 128)
    scratch_read = ops.get_universal_copy_atom(fx.BFloat16, 64)
    scratch_base = (wave % 4) * 16 * BN
    lane_group = lane // 16
    scale_buffer = None
    if const_expr(weight_quant_type == "ptpc"):
        scale_buffer = _weight_scale_buffer(p_w_scale, expert, N)
    pack = partial(
        pack_8x1_record, c=c, row_scale=row_scale, weight_quant_type=weight_quant_type
    )

    def shuffle_c_r2s2r(n, packed, row, half):
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
                + cshuffle_plane_offset(
                    oh * 8 + lane // 8, 2 * (atom_index % 2), ng // 2
                )
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

    store_c_r2g = partial(store_8x1_c_r2g, store_atom=store_atom)
    store_c_tile_r2g = partial(
        store_8x1_c_tile_r2g, shuffle_c_r2s2r=shuffle_c_r2s2r, store_c_r2g=store_c_r2g
    )
    acopy, afragments = _make_a_fragments(mm, a, BM, K_WIDTHS)
    save_state = partial(
        save_8x1_state,
        c=c,
        ptpc=weight_quant_type == "ptpc",
        first_unpacked=FIRST_UNPACKED,
    )
    run_tile = partial(
        run_k192_tile,
        n_tiles=NT,
        ptpc=weight_quant_type == "ptpc",
        read_b_g2r=read_b_g2r,
        store_b_r2s=store_b_r2s,
        read_b_s2r=read_b_s2r,
        read_scale_g2r=partial(
            read_8x1_scale_g2r,
            weight_quant_type=weight_quant_type,
            scale_buffer=scale_buffer,
            mm=mm,
            ops=ops,
        ),
        pack=pack,
        shuffle_c_r2s2r=shuffle_c_r2s2r,
        store_c_r2g=store_c_r2g,
        mma=partial(
            mfma_8x1_record, k_widths=K_WIDTHS, afragments=afragments, c=c, atom=atom
        ),
        clear=partial(clear_8x1_record, c=c),
        schedule_pack=schedule_pack,
        prepare_b_addresses=prepare_b_addresses,
    )

    def prologue():
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
        # P0=Q1=current H; P1=Q2=next N's L.
        return [read_b_g2r(0, 1), read_b_g2r(1, 0)]

    prepare_loop_state = partial(
        _prepare_8x1_loop_state,
        c=c,
        ptpc=weight_quant_type == "ptpc",
        first_unpacked=FIRST_UNPACKED,
        save_state=save_state,
    )

    # Pipeline: prologue keeps A resident, LDS=Q0, P0/P1=Q1/Q2; no prefetch without consumers.
    # N0 packs C0/C1, leaving FP32 C2/C3; the last two N tiles run after the loop.
    b_prefetch = prologue()
    _run_8x1_pipeline(
        b_prefetch,
        prepare_b_addresses(0),
        group,
        NT,
        2,
        FIRST_UNPACKED,
        run_tile,
        prepare_loop_state,
        save_state,
        pack,
        store_c_tile_r2g,
    )


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
    """Run one K192 N tile as L/H halves (q = 2*n + half), completing old C2/C3 in place.

    Returns the current N's packed C0/C1 and all four scales; previous_scales holds
    old scale2/scale3 only.
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
