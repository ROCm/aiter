# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Attention-residual (AttnRes) epilogue of the quantized **mesh** all-reduce.

The arithmetic is the one-shot epilogue's -- see
``one_shot_allreduce_attn_res`` for the model and for why each ``fastmath``
context is where it is. What differs is the schedule it hangs off and, because
of that, one design decision:

**the tile's rows are normed one at a time, not all together.** The mesh ends a
tile holding ``rows_per_tile`` whole rows (8 at every world size for the widths
this builds), and the RMSNorm epilogue reduces all of them in one
``rms_rstd`` call -- one barrier instead of eight. AttnRes cannot afford that:
a row's mix needs its ``nvb`` bank rows live *across* the block reduction, so
batching the tile would hold ``rows_per_tile * nvb`` widened bank atoms at once
-- 512 VGPRs at ``rows_per_tile=8, nvb=8``. Row at a time takes that to
``nvb * 8 = 64`` f32. The price is ``rows_per_tile`` block reductions per tile
instead of one; at prefill the epilogue reads ``rows_per_tile * nvb * hidden``
bf16 of bank per tile, so it is bandwidth-bound and the extra barriers are not
the critical path.

Row at a time is necessary but **not sufficient at the wide end**, and that is
the reason ``reread`` exists. The register budget is not a constant, and it is
not continuous in the block either: it is the occupancy staircase
``512 // ceil(block / 256)`` VGPRs, rounded down to the 8-register granule --
block 256/512 get 256 (166-257 used, nothing spills), block **640 and 768 get
168**, block 896 and 1024 get **128**. So even ``nvb * 8`` f32 spills at the
wide end, and the spill grows with ``nvb`` -- 40 / 56 / 64 / 93 / 107 at block
896 and 193 / 376 / **690** at block 1024, against only 8 / 20 / 28 / 43 at
block 640. Reading the bank a second time from HBM for
the mix instead of holding it caps the spill (57-59 at block 896, 133-143 at
block 1024) and past the crossover is cheaper than the scratch traffic it
replaces: at TP8, hidden 7168, nvb=8, T=8192 the epilogue goes
394.2 -> 308.7 us and the fused call goes from 1.02x of a plain all-reduce plus
the Triton aggregation to **1.18x**. Below the crossover it is a 10-13 % loss.
The *amount held* is ``nvb * atoms_per_row * 8`` f32, so the choice is keyed on
the row width ``block * atoms_per_row`` rather than on the block.
See ``_reread_default`` for the measured table and ``reread`` to pin it.

The ring is deliberately **not** built. Its register budget is easier
(``rows_per_chunk = 1`` at TP8), but at TP8 the ring is 3.7-5.3x slower than the
mesh on the all-reduce itself, so a fused ring build would be fusing onto the
wrong schedule.

Prototype-stage limitations, all deliberate and all host-visible:

* ``bank_nb`` is a **build constant** (the bank's row count per token), not a
  kernel argument. That is what lets this reuse the fused mesh launcher's
  existing signature unchanged -- ``q_ptr`` carries the bank, ``q_scale_ptr``
  the ``cw`` row and ``w_bias`` the score epsilon. A production build would add
  the four arguments properly, as the one-shot does.
* No ``low_latency`` build. A **padded** build is supported: ``h_pad`` widens
  the workgroup's row while every HBM operand keeps its ``hidden`` stride, so
  the bank needs a bounded descriptor per row (``_bank_buf``) where the native
  build uses one flat one. The reason to want it is the occupancy step --
  ``block`` falls with ``h_pad / atoms_per_row``, and this epilogue's cost is
  keyed on that step, not on the row width.
* MXFP4 output quantization is not combined with it.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from .one_shot_allreduce_attn_res import MAX_NVB, _block_reduce_many
from .one_shot_allreduce_rmsnorm import _LdsWindow
from .quick_allreduce_codec import _atom_bf16_to_f16
from .quick_allreduce_fusions import (
    ATOM_ELEMS,
    ATOM_I32,
    block_reduce_add,
    make_row_io,
    make_wave_partials,
    pack_bf16,
    quick_reduce_row_block_at,
    residual_add,
    resolve_h_pad,
    scale_by_weight,
)
from .quick_allreduce_rmsnorm import _atom_f16_to_f32
from .transport import (
    ATOMS,
    WAVE,
    _buffer_load,
    _buffer_ptr,
    _buffer_store,
    _to_sgpr_i64,
    atom_bf16_to_f32,
)


# ``_block_reduce_many`` and ``block_reduce_add`` each index a flat f32 array
# from 0, so the two reductions of one row take disjoint windows of the one
# allocation -- the same split the one-shot build uses, and for the same reason:
# the second's writes must not race a slow wave still reading the first's.


def _reread_default(row_block: int, nvb: int, atoms_per_row: int = 1) -> bool:
    """Should this build re-read the bank for the mix instead of holding it?

    The real rule is "re-read iff the holding build would spill", and that is
    now checkable -- ``vgpr_spill_count`` sits in the compiled artifact's
    ``gpu.binary`` ``kernel_metadata`` -- but only *after* compiling the build
    you are trying to choose. So this is the measured table instead, keyed on
    the **row width in atoms**, ``block * atoms_per_row`` (which is
    ``hidden // ATOM_ELEMS``, the same number at every world size and for every
    legal geometry of a width). It was keyed on ``block`` alone until N27, and
    that is only the same number when ``atoms_per_row == 1``: an
    ``atoms_per_row=2`` build holds twice the bank per thread at the same block,
    so it belongs in the wider row's cell. Measured consequence at TP4, T=8192,
    epilogue, hold vs re-read: hidden 10240 (block 640, apr 2, row 1280) at
    nvb=4 is **224.7 vs 136.6** -- the block key said hold and loses **1.64x**.

    * row 256 and 512 (hidden 2048 and 4096): at ``atoms_per_row=1``
      ``vgpr_count`` is 166-257 and **nothing spills at any nvb up to 8**, so
      the second HBM pass buys nothing and costs 10-13 % (hidden 4096, TP8,
      nvb=8: epilogue 146.7 and 269.3 us holding against 161.4 and 305.0
      re-reading, at T=8192/16384), and at TP2 hidden 4096 nvb=8 holding wins
      by 2.2-3.3x. **At ``atoms_per_row=2`` nvb=8 they cross, exactly as rows
      640-768 do** (N30), which is why this branch no longer stops at 640: the
      apr=2 holding build runs out of VGPRs and the compiler parks 69-80 values
      in the **AGPR** file, which the re-reading build never does. Measured at
      TP4, ``epi_us`` at T=8192, hold vs re-read: hidden 4096 (block 256, apr 2)
      **90.1/119.8 at nvb=5, 103.5/126.6 at nvb=6, 103.9/134.4 at nvb=7 and
      212.0/140.7 at nvb=8** -- holding by 1.22-1.33x up to nvb=7 and re-reading
      by **1.51x** at nvb=8, so the crossover is bracketed on both sides. Hidden
      2048 (block 128, apr 2) nvb=8 is **263.1/159.7, re-reading by 1.65x**,
      reproduced in three sessions to 1 % and at T=4096 as well. Worth 1.047x ->
      **1.244x** on the whole fused call at hidden 4096 nvb=8.
    * row 640 and 768 (hidden 5120 and 6144): at ``atoms_per_row=1`` the budget
      is **168**, one step of the occupancy staircase
      (``512 // ceil(block / 256)``, rounded to the 8-register granule) above
      896's 128, and the holding build spills only 8/20/28/43 (block 640) and
      14/30/34/51 (block 768) at nvb 4/5/6/8. **Through nvb=7 these widths take
      the narrow end's answer at every world size and in both geometries**:
      epilogue at T=8192, hold vs re-read, 118.4/188.4 (nvb=4), 140.3/222.4
      (nvb=5), 162.0/254.9 (nvb=6) at hidden 5120 TP8; 147.7/222.4 and
      167.8/253.5 at hidden 6144 TP8; 92.4/146.4 at TP4 and 72.8/113.7 at TP2
      for hidden 6144 nvb=6; and 85.9/140.9, 105.6/152.9, 142.2/160.6 at nvb
      5/6/7 for the TP4 hidden 6144 **apr=2** build. At nvb=5 that is the
      difference between a fused call of **1.34x** and one of 1.07x, so the
      boundary that matters is 896 and not 512.
      **nvb=8 is where the two geometries part**, and it is the geometry and not
      the world size: at row 768, hold vs re-read at T=8192 is 234.5/329.1 at
      TP8 and 109.4/189.3 at TP2 (both ``apr=1``, holding by ~40 %) but
      **385.4/191.3 at TP4** (``apr=2``, re-reading by 2.1x, reproduced at
      REPS=5). The ``apr=2`` build holds twice the bank per thread, so it alone
      crosses -- hence the ``atoms_per_row > 1`` term. Row 640 at nvb=8 is
      within 3-10 % either way at all three worlds.
    * row 896 (hidden 7168): ``vgpr_count`` is 128 and the holding build spills
      40/56/64/93/107 at nvb 1/2/4/6/8. Crossover measured at **nvb=5**
      (epilogue at T=8192: 184.6 vs 240.6 at nvb=4, 211.5 vs 208.2 at nvb=5,
      394.2 vs 308.7 at nvb=8).
    * row >= 1024 (hidden >= 8192): the holding build spills **193/376/690** at
      nvb 4/6/8 (hidden 16384) against 133/138/143 re-reading, so the crossover
      is at or below nvb=4 -- TP4 hidden 16384 (row 2048) T=4096 nvb=4: 244.0 vs
      **209.6**; TP8 hidden 8192 (row 1024) T=8192 nvb=4: 188.2 vs **168.2**;
      TP4 hidden 10240 (row 1280) T=8192 nvb=4: 224.7 vs **136.6**. At nvb=8 and
      hidden 8192 the two are within 3 % either way, so the rule costs nothing
      where it stops paying.

    Measured at hidden 2048 / 4096 / 5120 / 6144 / 7168 / 8192 / 10240 / 16384,
    at TP8, TP4 and TP2, with both ``atoms_per_row`` geometries at hidden 6144.
    Two known misses, both kept because no world-independent rule covers them:

    * TP4 hidden 5120 nvb=8 (row 640, apr 1) prefers re-reading by 9 % while the
      other two worlds prefer holding.
    * TP4 hidden 12288 nvb=4 (row 1536, apr 2) prefers **holding by 1.30x** on
      the epilogue (151.1 vs 196.5 us at T=8192, 239.8 vs 328.3 at T=16384,
      REPS=5) -- 6 % of the fused call, 1.312x holding against 1.240x
      re-reading. TP2 at the same cell prefers re-reading by 7 %, so the two
      worlds disagree. This is the one cell the ``>= 1024`` branch gets wrong:
      a full nvb=4 sweep of every legal row >= 1024 says re-read at TP2 at all
      five widths (by 8 % to 2.16x) and at TP4 at rows 1280 / 1792 / 2048 (by
      1.72x / 1.20x / 1.25x), with row 1024 a 3 % wash. Do not fit a third axis
      to it: the carrier is neither spill (monotone in width -- 104/112 holding
      and 52/66 re-reading at rows 1280/1536, TP4 -- while the preference
      flips) nor ``atoms_per_row`` (2 at every point of the TP4 sweep).
    """
    if row_block < 896:
        # Rows 256..768: the narrow end's answer through nvb=7 at every world
        # size and in both geometries. At nvb=8 the two geometries disagree
        # and the one-atom-per-row build is the one that wants to hold. This
        # branch used to stop at 640 with a categorical ``return False`` below
        # it; N30 measured rows 256 and 512 and they take the same condition,
        # so the two branches are one.
        return nvb >= 8 and atoms_per_row > 1
    if row_block >= 1024:
        return nvb >= 4
    return nvb >= 5


class MeshAttnResEpilogue:
    """AttnRes aggregation on the mesh's gathered tile (see the module doc)."""

    fusion = "attn_res"
    takes_operands = True
    mxfp4 = False
    low_latency = False

    def __init__(
        self,
        *,
        hidden: int,
        h_pad: int | None,
        block,
        world_size: int,
        nvb: int,
        bank_nb: int | None = None,
        write_bank: bool = True,
        out_norm: bool = True,
        reread: bool | None = None,
    ):
        hidden = int(hidden)
        # A padded build widens the *workgroup*, not the tensor: the wire tile
        # is ``rows_per_tile`` rows of ``h_pad`` while every HBM operand -- the
        # payload, the prefix pair, ``cw``, ``ow`` and the bank -- keeps its
        # true ``hidden`` stride, and the pad columns are bounded away per row
        # (``make_rowbuf_atom_row`` for the payload, ``_bank_buf`` for the
        # bank). Pad lanes then load 0, so they contribute nothing to either
        # block reduction, and their stores are dropped. ``inv_h`` stays
        # ``1 / hidden`` for exactly that reason.
        nvb = int(nvb)
        if not 1 <= nvb <= MAX_NVB:
            raise ValueError(f"nvb must be in [1, {MAX_NVB}], got {nvb}")
        self.hidden = hidden
        self.h_pad = resolve_h_pad(hidden, h_pad)
        self.nvb = nvb
        self.write_bank = bool(write_bank)
        self.out_norm = bool(out_norm)
        # Rows the bank holds per token: the nvb earlier states plus, when the
        # snapshot is on, the slot this call writes the prefix to.
        self.bank_nb = int(nvb + 1 if bank_nb is None else bank_nb)
        if self.bank_nb < nvb + (1 if self.write_bank else 0):
            raise ValueError(
                f"bank_nb={self.bank_nb} is too small for nvb={nvb}"
                f"{' with the snapshot on' if self.write_bank else ''}"
            )
        self.block, self.atoms_per_row = quick_reduce_row_block_at(
            self.h_pad, world_size, block
        )
        self.padded = self.h_pad != hidden
        self.reread = (
            _reread_default(self.block * self.atoms_per_row, nvb, self.atoms_per_row)
            if reread is None
            else bool(reread)
        )
        self.rank_atoms = ATOMS // world_size
        self.rows_per_tile = ATOMS // self.atoms_per_row
        self.rows_per_chunk = self.rank_atoms // self.atoms_per_row
        self.n_waves = self.block // WAVE
        self.row_stride_i32 = hidden // 2
        self.hbm_tile_bytes = self.rows_per_tile * self.row_stride_i32 * 4
        # One row at a time, so one row's worth of partials: the 2*(nvb+1) score
        # sums and then the output norm's own sum after them.
        self.n_score_sums = 2 * (nvb + 1)
        self.partial_floats = (
            (self.n_score_sums + 1) * self.n_waves if self.n_waves > 1 else 0
        )
        self._partials = (
            make_wave_partials(self.partial_floats) if self.partial_floats else None
        )

    @property
    def tag(self) -> str:
        tag = f"_attnres_h{self.hidden}_v{self.nvb}_nb{self.bank_nb}"
        # h_pad changes the geometry and which loads carry a bound, so a padded
        # build must not share a JIT key with the native one.
        if self.padded:
            tag += f"_p{self.h_pad}"
        if self.write_bank:
            tag += "_wb"
        if not self.out_norm:
            tag += "_nonorm"
        if self.reread:
            tag += "_rr"
        return tag

    def spec(self) -> dict:
        return {
            "fusion": self.fusion,
            "hidden": self.hidden,
            "h_pad": self.h_pad,
            "padded": self.padded,
            "mxfp4": False,
            "low_latency": False,
            "atoms_per_row": self.atoms_per_row,
            "rows_per_tile": self.rows_per_tile,
            "nvb": self.nvb,
            "bank_nb": self.bank_nb,
            "write_bank": self.write_bank,
            "out_norm": self.out_norm,
            "reread": self.reread,
        }

    def bind(self, **kwargs):
        return _BoundMeshAttnRes(self, **kwargs)


class _BoundMeshAttnRes:
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
        **_unused,
    ):
        """The fused mesh launcher's signature, reinterpreted (module doc):
        ``res_in``/``res_out`` are the prefix in and out, ``w``/``eps`` the
        output norm, ``w_bias`` the score epsilon, ``q_ptr`` the bank and
        ``q_scale_ptr`` the ``cw`` row."""
        self.epi = epi
        self.tid = tid
        self.eps = eps
        self.score_eps = w_bias
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
            # A partial last tile has rows past M; on a padded build their
            # per-row descriptors must not reach past the tensor. Unpadded, the
            # whole-payload descriptor already does this and the flag is unread.
            bound_rows=True,
        )
        self.io = io
        self.lds = None
        if epi._partials is not None:
            self.lds = allocator.allocate(epi._partials).peek().wave.ptr
        self.out_buf = io.operand(out_ptr)
        self.in_buf = io.operand(inp_ptr)
        self.prefix_in_buf = io.operand(res_in_ptr)
        self.prefix_out_buf = io.operand(res_out_ptr)

        # This thread's atom columns within one bf16 row, as i32 offsets.
        self.col_i32 = [
            (fx.Int32(a * epi.block) + tid) * fx.Int32(ATOM_I32)
            for a in range_constexpr(epi.atoms_per_row)
        ]
        # The bank is (M, bank_nb, hidden) bf16, outside the payload's window,
        # so it gets its own descriptor. Bounding it by the live row count is
        # what keeps a partial last tile's dead rows from reading or writing
        # past the tensor -- the same job ``nbytes`` does for the payload.
        #
        # A padded build cannot use one flat descriptor: its pad columns run
        # past the end of their own row, and a whole-tensor bound would read
        # the *next* bank row there instead of returning 0. So it bounds a
        # descriptor per row, in ``_bank_buf`` -- the bank's version of what
        # ``make_rowbuf_atom_row`` does for the payload.
        self.bank_ptr = _to_sgpr_i64(q_ptr)
        self.nbytes = nbytes
        self.bank_buf = None
        if const_expr(not epi.padded):
            self.bank_buf = _buffer_ptr(
                self.bank_ptr,
                T.i32,
                16,
                num_records_bytes=nbytes * fx.Int64(epi.bank_nb),
            )
        # cw is one fp32 (hidden,) row shared by every token: read once.
        cw_buf = _buffer_ptr(
            _to_sgpr_i64(q_scale_ptr),
            T.f32,
            16,
            num_records_bytes=fx.Int64(epi.hidden * 4),
        )
        self.cw = []
        for a in range_constexpr(epi.atoms_per_row):
            base = (fx.Int32(a * epi.block) + tid) * fx.Int32(ATOM_ELEMS)
            lo = _buffer_load(cw_buf, base, 4, fx.Float32)
            hi = _buffer_load(cw_buf, base + fx.Int32(4), 4, fx.Float32)
            self.cw.append(
                fx.Vector.from_elements(
                    [lo[e] for e in range_constexpr(4)]
                    + [hi[e] for e in range_constexpr(4)],
                    fx.Float32,
                )
            )

        self.ow_atoms = None
        if const_expr(epi.out_norm):
            w_buf = io.operand(w_ptr, records=fx.Int64(epi.hidden * 2))
            self.ow_atoms = [
                io.load(w_buf, fx.Int32(0), a)
                for a in range_constexpr(epi.atoms_per_row)
            ]

    # -- mesh hooks -----------------------------------------------------------

    def load_atom(self, tile, atom):
        return _atom_bf16_to_f16(self.io.load(self.in_buf, tile, atom))

    def stash_own(self, tile, atom, value):
        """As in the RMSNorm epilogue: park the raw fp16 bits in ``out`` and
        overwrite them below. The all-gather publish between the two drains
        ``vmcnt``, so the reload sees the store."""
        self.io.store(self.out_buf, tile, atom, value)

    def unstash_own(self, tile, atom):
        return self.io.load(self.out_buf, tile, atom)

    def prefetch(self, tile):
        """Not built: the hoist is the ``low_latency`` build's, which this
        prototype does not have."""
        return None

    def finish_tile(self, tile, gathered, pre=None):
        for r in range_constexpr(self.epi.rows_per_tile):
            self._agg_row(tile, r, gathered)

    # -- the row ---------------------------------------------------------------

    def _bank_row_i32(self, tile, r, j):
        """i32 offset of bank row *j* of the token at row *r* of *tile*."""
        token = tile * fx.Int32(self.epi.rows_per_tile) + fx.Int32(r)
        return (token * fx.Int32(self.epi.bank_nb) + fx.Int32(j)) * fx.Int32(
            self.epi.row_stride_i32
        )

    def _bank_buf(self, tile, r, j):
        """A padded build's descriptor over bank row *j* of row *r* of *tile*.

        Base is the row's first element, ``num_records`` its own ``hidden * 2``
        live bytes -- so this thread's pad columns sit past the end of the
        descriptor and load 0 / drop their stores, with no lane predicate. A
        row whose token is past ``M`` (the last tile is partial) gets 0 records
        instead, which is the row bound ``nbytes`` gives the payload.
        """
        epi = self.epi
        row_bytes = fx.Int64(epi.hidden * 2)
        token = tile * fx.Int32(epi.rows_per_tile) + fx.Int32(r)
        token64 = fx.Int64(token)
        tok_start = token64 * row_bytes
        base = self.bank_ptr + (token64 * fx.Int64(epi.bank_nb) + fx.Int64(j)) * (
            row_bytes
        )
        # ``select``, not ``if``: ``tok_start < nbytes`` is a runtime value.
        records = (tok_start < self.nbytes).select(row_bytes, fx.Int64(0))
        return _buffer_ptr(base, T.i32, 16, num_records_bytes=records)

    def _bank_row(self, tile, r, j):
        """Bank row *j* of row *r* of *tile*, widened to f32, one atom per
        column this thread owns."""
        if const_expr(self.epi.padded):
            buf = self._bank_buf(tile, r, j)
            return [
                atom_bf16_to_f32(
                    _buffer_load(buf, self.col_i32[a], ATOM_I32, fx.Int32)
                )
                for a in range_constexpr(self.epi.atoms_per_row)
            ]
        return [
            atom_bf16_to_f32(
                _buffer_load(
                    self.bank_buf,
                    self._bank_row_i32(tile, r, j) + self.col_i32[a],
                    ATOM_I32,
                    fx.Int32,
                )
            )
            for a in range_constexpr(self.epi.atoms_per_row)
        ]

    def _bank_store_row(self, tile, r, j, packed):
        """Write *packed* into bank row *j* of row *r* of *tile*."""
        if const_expr(self.epi.padded):
            buf = self._bank_buf(tile, r, j)
            for a in range_constexpr(self.epi.atoms_per_row):
                _buffer_store(buf, self.col_i32[a], packed[a])
            return
        off = self._bank_row_i32(tile, r, j)
        for a in range_constexpr(self.epi.atoms_per_row):
            _buffer_store(self.bank_buf, off + self.col_i32[a], packed[a])

    def _bank_rows(self, tile, r):
        return [self._bank_row(tile, r, j) for j in range_constexpr(self.epi.nvb)]

    def _row_dot_sq(self, row):
        """``dot(row, cw)`` and ``sum(row^2)`` over this thread's atoms.

        Reassociated for the reason ``one_shot_allreduce_attn_res._row_dot_sq``
        gives, and -- as there -- deliberately not contracted into a packed-FMA
        chain, which is faster and measurably less accurate.
        """
        dot = None
        sq = None
        for a in range_constexpr(len(row)):
            d = fx.Float32(
                (row[a] * self.cw[a]).reduce(ReductionOp.ADD, fastmath="reassoc")
            )
            s = fx.Float32(
                (row[a] * row[a]).reduce(ReductionOp.ADD, fastmath="reassoc")
            )
            dot = d if dot is None else dot + d
            sq = s if sq is None else sq + s
        return dot, sq

    def _agg_row(self, tile, r, gathered):
        epi = self.epi
        apr = epi.atoms_per_row
        idx = [r * apr + a for a in range_constexpr(apr)]
        inv_h = 1.0 / epi.hidden

        # prefix = bf16(prefix_in + bf16(all-reduce)), rounded before scoring.
        packed = pack_bf16(
            residual_add(
                [_atom_f16_to_f32(gathered[j]) for j in idx],
                [self.io.load(self.prefix_in_buf, tile, j) for j in idx],
            )
        )
        for a in range_constexpr(apr):
            self.io.store(self.prefix_out_buf, tile, idx[a], packed[a])
        if const_expr(epi.write_bank):
            self._bank_store_row(tile, r, epi.nvb, packed)
        pv = [atom_bf16_to_f32(p) for p in packed]

        # The widened bank rows stay live across the block reduction and feed
        # the mix below; this is the live range the whole design is about.
        # ``reread`` is the ablation of exactly that: score each bank row from a
        # load that dies immediately, then read the bank a second time for the
        # mix. It halves the live f32 and doubles the bank's HBM traffic.
        rows = None if const_expr(epi.reread) else self._bank_rows(tile, r)
        dots, sqs = [], []
        for j in range_constexpr(epi.nvb):
            # Emitted inside the loop when re-reading, so each row's load is
            # adjacent to its only two uses instead of spanning all nvb rows.
            row = self._bank_row(tile, r, j) if rows is None else rows[j]
            d, s = self._row_dot_sq(row)
            dots.append(d)
            sqs.append(s)
        d, s = self._row_dot_sq(pv)
        dots.append(d)
        sqs.append(s)

        lds = None
        if const_expr(epi.n_waves > 1):
            lds = _LdsWindow(self.lds, fx.Int32(0))
        totals = _block_reduce_many(dots + sqs, tid=self.tid, block=epi.block, lds=lds)
        n_rows = epi.nvb + 1
        with fx.fastmath("afn"):
            scores = [
                totals[k] * fmath.rsqrt(totals[n_rows + k] * inv_h + self.score_eps)
                for k in range_constexpr(n_rows)
            ]

        m = scores[0]
        for k in range_constexpr(1, n_rows):
            m = (scores[k] > m).select(scores[k], m)
        weights = [fmath.exp(s - m) for s in scores]
        denom = None
        for k in range_constexpr(epi.nvb):
            denom = weights[k] if denom is None else denom + weights[k]
        inv = fx.Float32(1.0) / (denom + weights[epi.nvb])

        # Under ``reread`` the bank comes back from HBM here, one row at a time,
        # so only the running accumulator and one row are live.
        accs = [None] * apr
        for k in range_constexpr(epi.nvb):
            row = self._bank_row(tile, r, k) if rows is None else rows[k]
            wk = weights[k] * inv
            for a in range_constexpr(apr):
                with fx.fastmath("contract"):
                    term = row[a] * wk
                    accs[a] = term if accs[a] is None else accs[a] + term
        mix = []
        for a in range_constexpr(apr):
            with fx.fastmath("contract"):
                pterm = pv[a] * (weights[epi.nvb] * inv)
                mix.append(pterm if accs[a] is None else pterm + accs[a])

        if const_expr(epi.out_norm):
            local = None
            for a in range_constexpr(apr):
                s = fx.Float32(
                    (mix[a] * mix[a]).reduce(ReductionOp.ADD, fastmath="reassoc")
                )
                local = s if local is None else local + s
            lds2 = None
            if const_expr(epi.n_waves > 1):
                lds2 = _LdsWindow(
                    self.lds, fx.Int32(epi.n_score_sums * epi.n_waves)
                )
            total = block_reduce_add(
                [local], tid=self.tid, block=epi.block, lds=lds2
            )[0]
            scale = fx.Float32(1.0) / fmath.sqrt(total * inv_h + self.eps)
            out = scale_by_weight(mix, scale, self.ow_atoms)
        else:
            out = pack_bf16(mix)
        for a in range_constexpr(apr):
            self.io.store(self.out_buf, tile, idx[a], out[a])

        if const_expr(epi.n_waves > 1 and not epi.out_norm):
            # With the output norm on, its own reduction's barrier is what
            # separates this row's reads of the score partials from the next
            # row's writes of them. Without it there is nothing in between.
            gpu.barrier()


__all__ = ["MeshAttnResEpilogue"]
