# SPDX-License-Identifier: Apache-2.0
"""GEMM2 output policies for the fused TP megakernel."""

from dataclasses import dataclass

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_d
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.expr import arith, range_constexpr
from flydsl.expr.typing import T

from .... import vector


@dataclass(frozen=True)
class RouteOutputEpilogue:
    """Publish one BF16 or fixed-scale FP8 route from the GEMM epilogue."""

    row_width: int
    fp8_fixed: bool
    device_coherent: bool

    @property
    def row_bytes(self) -> int:
        return self.row_width if self.fp8_fixed else self.row_width * 2

    def cshuffle_layout(self, tile_n: int) -> tuple[int, int]:
        if self.fp8_fixed and self.device_coherent and tile_n % 256 == 0:
            return 16, 16
        return min(tile_n // 32, 8), 32

    def store(
        self,
        *,
        row_ctx,
        col_g0,
        frag,
        e_vec: int,
        idx_to_llvm_ptr,
    ):
        _, row_byte_base, _ = row_ctx
        element_bytes = 1 if self.fp8_fixed else 2
        output_address = row_byte_base + col_g0 * arith.constant(
            element_bytes, index=True
        )
        if not self.fp8_fixed:
            output_pointer = idx_to_llvm_ptr(output_address)
            if self.device_coherent:
                if e_vec * element_bytes != 16:
                    raise RuntimeError(
                        "coherent BF16 route store requires one 16-byte vector"
                    )
                packed = vector.bitcast(T.vec(4, T.i32), frag)
                llvm_d.InlineAsmOp(
                    None,
                    [output_pointer, packed],
                    "global_store_dwordx4 $0, $1, off sc1",
                    "v,v",
                    has_side_effects=True,
                )
            else:
                raw = frag._value if hasattr(frag, "_value") else frag
                llvm_d.StoreOp(
                    raw,
                    output_pointer,
                    alignment=e_vec * element_bytes,
                    nontemporal=True,
                )
            return

        fragment = fx.Vector(frag)
        values = [
            fragment[element].to(fx.Float32) for element in range_constexpr(e_vec)
        ]
        packed_words = []
        for word in range_constexpr(e_vec // 4):
            element = word * 4
            packed = fx.Int32(0)
            packed = fx.rocdl.cvt_pk_fp8_f32(
                T.i32,
                values[element],
                values[element + 1],
                packed,
                0,
            )
            packed = fx.rocdl.cvt_pk_fp8_f32(
                T.i32,
                values[element + 2],
                values[element + 3],
                packed,
                1,
            )
            raw = packed._value if hasattr(packed, "_value") else packed
            packed_words.append(raw)
            if not self.device_coherent:
                llvm_d.StoreOp(
                    raw,
                    idx_to_llvm_ptr(
                        output_address + arith.constant(word * 4, index=True)
                    ),
                    alignment=4,
                    nontemporal=True,
                )

        if self.device_coherent:
            word_count = e_vec // 4
            if word_count not in (1, 2, 4):
                raise RuntimeError(
                    "coherent FP8 route store requires 1, 2, or 4 dwords"
                )
            packed_store = (
                packed_words[0]
                if word_count == 1
                else vector.from_elements(T.vec(word_count, T.i32), packed_words)
            )
            store_suffix = "" if word_count == 1 else f"x{word_count}"
            llvm_d.InlineAsmOp(
                None,
                [idx_to_llvm_ptr(output_address), packed_store],
                f"global_store_dword{store_suffix} $0, $1, off sc1",
                "v,v",
                has_side_effects=True,
            )


@dataclass(frozen=True)
class Gemm2TPComposition:
    """TP-pipeline options consumed by the shared GEMM2 emitter."""

    compose_entry: object | None = None
    n_tile_range: tuple[int, int] | None = None
    block_threads: int = 256
    persistent_groups: int | None = None
    output_epilogue: RouteOutputEpilogue | None = None
    b_cache_modifier: int = 0

    @staticmethod
    def emit_final_sync(iteration, one, iteration_count):
        final_iteration = scf.IfOp(
            arith.cmpi(
                CmpIPredicate.eq,
                iteration + one,
                iteration_count,
            )
        )
        with ir.InsertionPoint(final_iteration.then_block):
            fx.rocdl.s_waitcnt(0)
            scf.YieldOp([])
