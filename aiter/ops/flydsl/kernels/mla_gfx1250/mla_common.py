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
