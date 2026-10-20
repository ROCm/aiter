# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Helpers shared by the gfx1250 MLA kernels."""

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

_XOR16_SEL_LO = 0x76543210
_XOR16_SEL_HI = 0xFEDCBA98 - (1 << 32)


def _instruction_prefetch(num_pages):
    from flydsl._mlir.dialects import llvm

    lines = [
        f"s_prefetch_inst_pc_rel 0x{page * 0x1000:x}, $0, 31"
        for page in range(num_pages)
    ]
    llvm.inline_asm(
        None,
        [fx.Int32(0).ir_value()],
        "\n".join(lines),
        "s",
        has_side_effects=True,
    )


def _xor16_f32(value):
    sel_lo = fx.Int32(_XOR16_SEL_LO).ir_value()
    sel_hi = fx.Int32(_XOR16_SEL_HI).ir_value()
    src = value.ir_value()
    return fx.Float32(rocdl.permlanex16(T.f32, src, src, sel_lo, sel_hi, False, False))


def _concat_wmma_operand(chunks):
    v01 = chunks[0].shuffle(chunks[1], list(range(8)))
    v23 = chunks[2].shuffle(chunks[3], list(range(8)))
    return v01.shuffle(v23, list(range(16)))


def _concat_wmma_operand_k64(chunks):
    return chunks[0].shuffle(chunks[1], list(range(8)))


def _concat_ds_tr8_b64(chunks):
    v01 = chunks[0].shuffle(chunks[1], list(range(4)))
    v23 = chunks[2].shuffle(chunks[3], list(range(4)))
    return v01.shuffle(v23, list(range(8)))


def _rmem_i32(n, value):
    fragment = fx.make_rmem_tensor(n, fx.Int32)
    fragment.store(value)
    return fragment


def make_fp8_wmma_atom(k):
    return fx.make_mma_atom(
        fx.rocdl.WMMA(
            16,
            16,
            k,
            fx.Float8E4M3FN,
            fx.Float32,
        )
    )


def _pack_fp8x4(values, base):
    packed = rocdl.cvt_pk_fp8_f32(
        T.i32,
        values[base],
        values[base + 1],
        fx.Int32(0),
        0,
    )
    return fx.Int32(
        rocdl.cvt_pk_fp8_f32(
            T.i32,
            values[base + 2],
            values[base + 3],
            packed,
            1,
        )
    )


def _dwordx4_iter(ptr):
    return fx.recast_iter(
        fx.PointerType.get(fx.Int32.ir_type, ptr.memspace, 16),
        ptr,
    )


def make_global_load_b128():
    layout = fx.make_layout(4, 1)
    atom = fx.make_copy_atom(fx.UniversalCopy(128), fx.Int32)

    def load(base, dword_offset):
        rmem = fx.make_rmem_tensor(layout, fx.Int32)
        source = fx.Tensor(fx.make_view(fx.add_offset(base, dword_offset), layout))
        fx.copy_atom_call(atom, source, rmem)
        return rmem.load()

    return load


BLOCK_THREADS = 128
WAVE_SIZE = 32
NUM_WAVES = BLOCK_THREADS // WAVE_SIZE
HEADS_PER_GROUP = 16
MAX_GROUPS_PER_WAVE = 2
SUPPORTED_NUM_Q_HEADS = (16, 32, 64, 96, 128)

PLANNER_PACKED_Q_BUDGET = 128

QK_NOPE_HEAD_DIM = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM = QK_NOPE_HEAD_DIM
PV_D_TILES = V_HEAD_DIM // 16


def wave_layout(num_q_heads, max_seqlen_q):
    """(q_tile_rows, head_groups, groups_per_wave, m_waves, dv_waves)."""
    if num_q_heads not in SUPPORTED_NUM_Q_HEADS:
        raise ValueError(
            f"num_q_heads: expected one of {list(SUPPORTED_NUM_Q_HEADS)}, "
            f"got {num_q_heads}"
        )
    q_tile_rows = 1 if num_q_heads * 2 > PLANNER_PACKED_Q_BUDGET else max_seqlen_q
    if q_tile_rows > 4 or num_q_heads * q_tile_rows > PLANNER_PACKED_Q_BUDGET:
        raise ValueError(
            f"num_q_heads={num_q_heads} supports max_seqlen_q up to "
            f"{min(4, PLANNER_PACKED_Q_BUDGET // num_q_heads)}, got {max_seqlen_q}"
        )
    head_groups = num_q_heads // HEADS_PER_GROUP
    q_slots = 1 if q_tile_rows == 1 else 2 if q_tile_rows == 2 else 4
    groups = head_groups * q_slots
    groups_per_wave = MAX_GROUPS_PER_WAVE if groups > NUM_WAVES else 1
    # A wave's groups share one query position.
    assert head_groups % groups_per_wave == 0
    m_waves = groups // groups_per_wave
    dv_waves = NUM_WAVES // m_waves
    if dv_waves == 0 or PV_D_TILES % dv_waves != 0:
        raise ValueError(
            f"num_q_heads={num_q_heads} with max_seqlen_q={max_seqlen_q} needs "
            f"{m_waves} m-waves, which does not tile {NUM_WAVES} waves x "
            f"{PV_D_TILES} output tiles"
        )
    return q_tile_rows, head_groups, groups_per_wave, m_waves, dv_waves


def _unhoisted(value):
    from flydsl._mlir.dialects import llvm

    return fx.Int32(
        llvm.inline_asm(
            T.i32,
            [fx.Int32(value).ir_value()],
            "v_mov_b32 $0, $1",
            "=v,v",
            has_side_effects=True,
        )
    )


def _softmax_all(
    softmax_group, ctx, qk_accs, running_max, running_sum, valid_count, mask_bounds, out
):
    results = [[] for _ in range(ctx.G)]
    yield from _round_robin(
        [
            softmax_group(
                ctx,
                qk_accs[g],
                running_max[g],
                running_sum[g],
                valid_count,
                mask_bounds,
                results[g],
            )
            for g in range(ctx.G)
        ]
    )
    rescales = results[0][3]
    for g in range(1, ctx.G):
        rescales = rescales | results[g][3]
    any_rescale = fx.Int32(rocdl.ballot(T.i32, rescales.ir_value())) != fx.Int32(0)
    out.extend(
        [r[0] for r in results]
        + [r[1] for r in results]
        + [r[2] for r in results]
        + [any_rescale.select(fx.Int32(1), fx.Int32(0))]
        + [r[4] for r in results]
    )


def _round_robin(steppers):
    active = list(steppers)
    while active:
        for stepper in tuple(active):
            if next(stepper, _DONE) is _DONE:
                active.remove(stepper)
            else:
                yield


def _drain(stepper):
    for _ in stepper:
        pass


_DONE = object()


def _min_u32(lhs, rhs):
    from flydsl._mlir.dialects import arith

    return fx.Int32(arith.minui(fx.Int32(lhs).ir_value(), fx.Int32(rhs).ir_value()))


def _wait_scalars(values):
    from flydsl._mlir.dialects import llvm

    llvm.inline_asm(
        None,
        [fx.Int32(value).ir_value() for value in values],
        "",
        ",".join(["s"] * len(values)),
        has_side_effects=True,
    )
