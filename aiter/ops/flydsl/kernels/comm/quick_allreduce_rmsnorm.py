# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Residual-add + RMSNorm epilogues of the quantized (mesh, ring) all-reduce.

The hooks are documented in ``quick_allreduce_epilogue``. Both schedules size
the block so that ``atoms_per_row`` 16 B atoms are one token row (see
``quick_allreduce_fusions.quick_reduce_row_block``): the mesh ends a tile
holding ``rows_per_tile`` whole rows, and every chunk the ring completes is
``rows_per_chunk`` whole rows, so the norm runs inside the op that receives it
rather than waiting for the tile to be reassembled.

The fused operands -- input, residual, weight and the final output -- are read
as raw bf16 atoms through ``quick_allreduce_fusions.RowIO`` rather than the
plain path's ``_payload_io``. At ``h_pad != hidden`` the wire tile is wider than
the HBM row, so ``_payload_io``'s single whole-payload bound would let a pad
lane read the next row instead of zero -- which would then ride the wire into
``residual_out``.
"""

import flydsl.expr as fx
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.typing import ReductionOp, T, as_ir_value

from .. import buffer_ops
from ..quant_utils import emit_mx_e8m0_scale
from .quick_allreduce_codec import _atom_bf16_to_f16
from .quick_allreduce_fusions import (
    MXFP4_GROUP,
    make_row_io,
    make_wave_partials,
    mxfp4_scale_shape,
    pack_bf16,
    quick_reduce_row_block_at,
    residual_add,
    resolve_h_pad,
    rms_rstd,
    scale_by_weight,
)
from .transport import ATOMS, WAVE


def _atom_f16_to_f32(atom):
    """Packed fp16 -> 8 f32. A widening move; exact, no rounding."""
    return fx.Vector(atom).bitcast(fx.Float16).to(fx.Float32)


class _QuickReduceRMSNorm:
    fusion = "rmsnorm"
    takes_operands = True

    def __init__(
        self,
        *,
        hidden: int,
        h_pad: int | None,
        block,
        world_size: int,
        mxfp4: bool = False,
    ):
        hidden = int(hidden)
        self.hidden = hidden
        self.h_pad = resolve_h_pad(hidden, h_pad)
        # Also quantize ``out`` to MXFP4 (packed fp4x2 + shuffled e8m0 scales),
        # the layout ``per_1x32_f4_quant_hip(out, shuffle=True)`` writes.
        self.mxfp4 = bool(mxfp4)
        if self.mxfp4 and (self.h_pad != hidden or hidden % MXFP4_GROUP):
            raise ValueError(
                f"mxfp4 needs an unpadded hidden that is a multiple of "
                f"{MXFP4_GROUP}, got hidden={hidden} (h_pad={self.h_pad})"
            )
        # An explicit block picks among the widths this hidden dim admits; None
        # keeps the widest.
        self.block, self.atoms_per_row = quick_reduce_row_block_at(
            self.h_pad, world_size, block
        )
        self.padded = self.h_pad != hidden
        self.rank_atoms = ATOMS // world_size
        self.rows_per_tile = ATOMS // self.atoms_per_row
        self.rows_per_chunk = self.rank_atoms // self.atoms_per_row
        self.n_waves = self.block // WAVE
        # i32 between one token row and the next *in HBM*, and the HBM bytes a
        # whole tile spans. The wire tile is rows_per_tile rows of h_pad, the
        # HBM footprint rows_per_tile rows of hidden; the host derives
        # num_tiles from ``hbm_tile_bytes``.
        self.row_stride_i32 = hidden // 2
        self.hbm_tile_bytes = self.rows_per_tile * self.row_stride_i32 * 4
        # Per-wave partials for the sum of squares, one slot per row one norm
        # call reduces. Allocated separately from the pack staging, so it
        # cannot alias a packet the fanout is still reading.
        n_rows = self._rows_per_norm()
        self.partial_floats = n_rows * self.n_waves if self.n_waves > 1 else 0
        self._partials = (
            make_wave_partials(self.partial_floats) if self.partial_floats else None
        )

    @property
    def tag(self) -> str:
        # h_pad changes both the geometry and which loads carry a mask, so a
        # padded build must not share a JIT key with the plain one.
        tag = f"_rms_h{self.hidden}"
        if self.padded:
            tag += f"_p{self.h_pad}"
        if self.mxfp4:
            tag += "_mxfp4"
        return tag

    def spec(self) -> dict:
        return {
            "fusion": self.fusion,
            "hidden": self.hidden,
            "h_pad": self.h_pad,
            "padded": self.padded,
            "mxfp4": self.mxfp4,
            "atoms_per_row": self.atoms_per_row,
            "rows_per_tile": self.rows_per_tile,
        }


class _BoundQuickReduceRMSNorm:
    def __init__(
        self,
        epi,
        *,
        tid,
        nbytes,
        num_tiles,
        inp_ptr,
        out_ptr,
        res_in_ptr,
        res_out_ptr,
        w_ptr,
        eps,
        w_bias,
        q_ptr,
        q_scale_ptr,
        allocator,
    ):
        self.epi = epi
        self.tid = tid
        self.eps = eps
        self.w_bias = w_bias
        self.nbytes = nbytes
        if epi.mxfp4:
            self._bind_mxfp4(q_ptr, q_scale_ptr)
        io = make_row_io(
            tid=tid,
            nbytes=nbytes,
            num_tiles=num_tiles,
            block=epi.block,
            atoms_per_row=epi.atoms_per_row,
            rows_per_tile=epi.rows_per_tile,
            row_stride_i32=epi.row_stride_i32,
            padded=epi.padded,
            hidden=epi.hidden,
            # A partial last tile has rows past M; their descriptors must not
            # reach past the tensor.
            bound_rows=True,
        )
        self.io = io
        self.sq_lds = None
        if epi._partials is not None:
            # A second region from the kernel's allocator -- FlyDSL permits one
            # per kernel. Bound once, at kernel scope: the allocator is static,
            # and one reached from inside the tile loop would emit an LDS symbol
            # per visit.
            self.sq_lds = allocator.allocate(epi._partials).peek().wave.ptr
        # residual in/out share the payload's (M, hidden) bf16 shape, so they
        # ride the same addressing, as does the final output row. The gain is
        # one (hidden,) row -- ``atoms_per_row`` atoms at tile 0; a padded build
        # reads it through the per-row descriptor at row 0, an unpadded one
        # bounds a one-row buffer tensor at the true width (the whole mask for
        # this operand, since nothing lies past it).
        self.out_buf = io.operand(out_ptr)
        self.res_in_buf = io.operand(res_in_ptr)
        self.res_out_buf = io.operand(res_out_ptr)
        w_buf = io.operand(w_ptr, records=fx.Int64(epi.hidden * 2))
        self.in_buf = io.operand(inp_ptr)
        # One row shared by every token, so the gain is read once here rather
        # than once per tile.
        self.w_atoms = [
            io.load(w_buf, fx.Int32(0), a) for a in range_constexpr(epi.atoms_per_row)
        ]

    def _bind_mxfp4(self, q_ptr, q_scale_ptr):
        epi = self.epi
        self.scale_n = mxfp4_scale_shape(1, epi.hidden)[1]
        self.n_rows = self.nbytes // fx.Int64(epi.hidden * 2)
        self.q_rsrc = buffer_ops.create_buffer_resource_from_addr(
            q_ptr, num_records_bytes=self.nbytes // 4
        )
        self.q_scale_rsrc = buffer_ops.create_buffer_resource_from_addr(
            q_scale_ptr,
            num_records_bytes=(self.n_rows + 255) // 256 * fx.Int64(256 * self.scale_n),
        )
        # Lane part of the shuffled scale index for group ``tid // 4`` of an
        # atom; the atom's group base is a multiple of 8, so the row and atom
        # parts fold into soffset.
        g = self.tid // fx.Int32(MXFP4_GROUP // 8)
        self.scale_lane = (g // 8) * 256 + (g % 4) * 64 + ((g % 8) // 4) * 2

    def _store_mxfp4(self, tile, row_in_tile, a, out_atom):
        """MXFP4-quantize atom *a* of the row *row_in_tile* of ``out``.

        A 32-element group spans four consecutive lanes. The round-up e8m0
        scale and the hardware scaled convert match ``per_1x32_f4_quant_hip``
        bit for bit. Rows past the payload are dropped by the mask.
        """
        epi = self.epi
        yb = fx.Vector(out_atom).bitcast(fx.BFloat16)
        amax = abs(yb.to(fx.Float32)).reduce(ReductionOp.MAX)
        for off in range_constexpr(2):
            amax = fx.maxnumf(amax, amax.shuffle_xor(1 << off, WAVE))
        e8m0 = fx.Int32(emit_mx_e8m0_scale(amax))
        scale = as_ir_value((e8m0 << 23).bitcast(fx.Float32))
        words = fx.Vector(out_atom)
        pk = as_ir_value(fx.Int32(0))
        for j in range_constexpr(4):
            pk = rocdl.cvt_scalef32_pk_fp4_bf16(
                T.i32,
                pk,
                as_ir_value(
                    fx.Vector.from_elements([words[j]], fx.Int32).bitcast(fx.BFloat16)
                ),
                scale,
                j,
            )
        row = tile * fx.Int32(epi.rows_per_tile) + fx.Int32(row_in_tile)
        live = fx.Int64(row) < self.n_rows
        buffer_ops.buffer_store(
            fx.Int32(pk),
            self.q_rsrc,
            self.tid,
            mask=live,
            soffset_bytes=row * fx.Int32(epi.hidden // 2) + fx.Int32(a * epi.block * 4),
        )
        # The group's four lanes store the same byte.
        scale_row = (
            (row // fx.Int32(32)) * fx.Int32(32 * self.scale_n)
            + (row % fx.Int32(16)) * fx.Int32(4)
            + (row % fx.Int32(32)) // fx.Int32(16)
        )
        groups_per_atom = epi.block * 8 // MXFP4_GROUP
        buffer_ops.buffer_store(
            arith.trunci(T.i8, as_ir_value(e8m0)),
            self.q_scale_rsrc,
            self.scale_lane,
            mask=live,
            soffset_bytes=scale_row + fx.Int32(a * groups_per_atom // 8 * 256),
        )

    def load_atom(self, tile, atom):
        """The input reaches ``inp`` through the per-row bounded route, so a
        padded row's pad lanes come back zero; bf16 -> fp16 for the codec."""
        return _atom_bf16_to_f16(self.io.load(self.in_buf, tile, atom))

    def _norm_rows(self, tile, base, n_rows, values, residual):
        """Residual add + RMSNorm over *n_rows* whole rows starting at atom
        *base* of *tile*.

        *values* are the reduced fp16 atoms of those rows, *residual(j)* the
        raw residual atom ``base + j``. All rows go through one ``rms_rstd``
        call, which is one barrier rather than one per row.
        """
        epi = self.epi
        apr = epi.atoms_per_row
        rows = []
        for r in range_constexpr(n_rows):
            idx = [r * apr + a for a in range_constexpr(apr)]
            xs = residual_add(
                [_atom_f16_to_f32(values[j]) for j in idx],
                [residual(j) for j in idx],
            )
            rows.append(xs)
            # residual_out is final here, so it is issued ahead of the
            # reduction rather than behind its barrier.
            packed = pack_bf16(xs)
            for a in range_constexpr(apr):
                self.io.store(self.res_out_buf, tile, base + idx[a], packed[a])

        rstds = rms_rstd(
            rows, self.eps, epi.hidden, tid=self.tid, block=epi.block, lds=self.sq_lds
        )
        for r in range_constexpr(n_rows):
            outs = scale_by_weight(rows[r], rstds[r], self.w_atoms, self.w_bias)
            for a in range_constexpr(apr):
                self.io.store(self.out_buf, tile, base + r * apr + a, outs[a])
                if epi.mxfp4:
                    self._store_mxfp4(tile, base // apr + r, a, outs[a])
        if self.sq_lds is not None:
            # The next call reuses these slots, and nothing else between here
            # and there is guaranteed to be a barrier -- the mesh's super-tile
            # store loop runs back-to-back, the ring's last op has no fanout --
            # so this is what separates its writes from these reads.
            gpu.barrier()


class MeshRMSNormEpilogue(_QuickReduceRMSNorm):
    """The mesh ends a tile holding every atom of it: ``rows_per_tile``
    independent rows, each reduced inside this block."""

    def _rows_per_norm(self):
        return self.rows_per_tile

    def bind(self, **kwargs):
        return _BoundMeshRMSNorm(self, **kwargs)


class _BoundMeshRMSNorm(_BoundQuickReduceRMSNorm):
    def stash_own(self, tile, atom, value):
        """Park the raw fp16 bits of our reduced chunk in ``out`` for
        ``unstash_own`` -- exact, where a bf16 store would round -- and the
        epilogue then overwrites them. The all-gather publish between the two
        drains ``vmcnt``, so the reload sees the store."""
        self.io.store(self.out_buf, tile, atom, value)

    def unstash_own(self, tile, atom):
        return self.io.load(self.out_buf, tile, atom)

    def finish_tile(self, tile, gathered):
        self._norm_rows(
            tile,
            0,
            self.epi.rows_per_tile,
            gathered,
            lambda j: self.io.load(self.res_in_buf, tile, j),
        )


class RingRMSNormEpilogue(_QuickReduceRMSNorm):
    """The ring completes a chunk at a time, and a chunk is ``rows_per_chunk``
    whole rows. The epilogue runs *after* the chunk has been forwarded, so no
    norm arithmetic sits between a receive and the send that unblocks the
    successor."""

    def _rows_per_norm(self):
        return self.rows_per_chunk

    def bind(self, **kwargs):
        return _BoundRingRMSNorm(self, **kwargs)


class _BoundRingRMSNorm(_BoundQuickReduceRMSNorm):
    def begin_chunk(self, tile, chunk, recv):
        """This chunk's residual atoms, issued before the payload arrives.

        Plain HBM with no dependence on any peer, so the load retires while the
        receive and the dequantize are still running rather than stalling the
        epilogue on a cold read after them. ``recv[0]``; the completed atoms
        follow it.
        """
        base = chunk * self.epi.rank_atoms
        recv.append(
            [
                self.io.load(self.res_in_buf, tile, base + j)
                for j in range_constexpr(self.epi.rank_atoms)
            ]
        )

    def complete(self, tile, atom, value, recv):
        """Held for the epilogue, which needs the whole row."""
        recv.append(value)

    def end_chunk(self, tile, chunk, recv):
        res_atoms = recv[0]
        self._norm_rows(
            tile,
            chunk * self.epi.rank_atoms,
            self.epi.rows_per_chunk,
            recv[1:],
            lambda j: res_atoms[j],
        )
