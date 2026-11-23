# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""MXFP8-weight tile bodies for the DeepSeek-V4.1 shared expert (default off).

Nothing on a served path imports this module: the MegaMoE kernels, their names and their compile-cache keys are
what they were. It holds the two tile bodies a fused shared expert needs, each also as a standalone kernel so it
can be checked against the dense launches it stands in for:

  GEMM1  rows [M, K] MXFP8 (1x32 E8M0, row-major) x ``[gate | up]`` weight [2 * D, K] MXFP8 (shuffle_weight (16, 16),
         32x32 E8M0 blocks, row-major [2 * D / 32, K / 32]) -> the activation pair ``silu_mul_mxfp8`` writes:
         E4M3 [M, D] and row-major E8M0 [M, D / 32].
  GEMM2  that pair x the down weight [N, D] MXFP8 (same formats) -> BF16 rows [M, N].

Both reuse the routed tile's loop (gemm1.do_tile_la: 32-row x 256-column tiles, 256-K steps, B one step ahead,
16x16x128 scaled MFMA, zero-initialized accumulators, K ascending) with FP8 B loaders, so every accumulator sees
the 128-K MFMA steps in the order of the dense ``sk1`` kernels. ``ksplit=2`` runs the two K halves into separate
accumulators and sums them as ``0 + p0 + p1``, the association of the dense split-K 2 kernels (gate_up at 33..64
rows). The GEMM1 epilogue rounds gate / up to BF16 and then runs AITER's transcription of ``silu_mul_mxfp8``
(aiter.ops.flydsl.kernels.silu_mul_quant_epilogue) in that kernel's float mode.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_ops
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.typing import as_ir_value as _raw

from .. import silu_mul_quant_epilogue as sq
from ..mxfp4_gemm_common import lds_typed_ptr
from ..tensor_shim import _run_compiled, buf_copy_load, buf_copy_store, ptr_buf_tensor
from .gemm1 import _LdsF32View, do_tile_la
from .gemm_util import (
    _PACK,
    AS2RLoader,
    AScaleLoader,
    ATileLoader,
    MfmaScaleGU,
    wait_lds_barrier,
)

__all__ = [
    "PARTIAL_UNIT_BYTES",
    "Bf16RowsEpilogue",
    "RefActQuantEpilogue",
    "build_shared_half_tiles",
    "build_shared_tile",
    "compile_shared_gemm1",
    "compile_shared_gemm2",
    "dense_gate_up_ksplit",
    "shared_gemm1",
    "shared_gemm2",
]

SORT_BLOCK_M = 32
TILE_N = 256
NUM_WAVES = 4
K_STEP = 256
THREADS = NUM_WAVES * 64
M_REPEAT = SORT_BLOCK_M // 16
NUM_ACC_N = TILE_N // NUM_WAVES // 16
# build_shared_half_tiles: bytes of one tile's first-half partials in memory (16 bytes per accumulator and thread).
PARTIAL_UNIT_BYTES = THREADS * M_REPEAT * NUM_ACC_N * 16
# A store index past every buffer bound (megamoe's spelling of a dropped store).
_OOB = 0x40000000
# MODE's FP32 denormal field of a FlyDSL kernel (denormals kept); silu_mul_quant_epilogue flushes them.
_FP32_DENORM_KEPT = 3


def f8_weight_group(row, k128, half, k_lane, row_lane, k_dim):
    """16-byte group index, in a shuffle_weight(., (16, 16)) FP8 weight of ``k_dim`` columns, of the chunk
    W[row + row_lane][128 k128 + 64 half + 16 k_lane ..+16] (``row`` a multiple of 16). Lane ``16 k_lane +
    row_lane`` of the MFMA B operand of 128-K step ``k128`` is the chunks of half 0 and 1.
    """
    return row * (k_dim // 16) + (k128 * 2 + half) * 64 + k_lane * 16 + row_lane


def gate_up_row(row_base, ni, half):
    """First weight row of a wave's accumulator ``ni``: ``row_base + 16 ni`` of a plain weight; of a
    ``[gate | up]`` weight (``half`` = its gate rows) the gate (ni even) or up (ni odd) rows of activation columns
    ``row_base // 2 + 16 (ni // 2)``."""
    if half:
        return row_base // 2 + (ni // 2) * 16 + (ni % 2) * half
    return row_base + ni * 16


def scale32_dword(row32, kstep, ksub, k_dim):
    """Dword index, in a row-major [rows / 32, k_dim / 32] E8M0 scale, of the four K-group scales of scale row
    ``row32`` at 128-K step ``ksub`` of 256-K step ``kstep`` (byte g = K group g)."""
    return row32 * (k_dim // 32 // 4) + kstep * 2 + ksub


def _static_gate_up_ksplit(rows):
    """The gate_up split of AITER's tuned dsv41 table as of d29c94315 (N 4608, K 5120, 32x32 scales): its 48- and
    64-row tiers are split-K 2 kernels, every other tier up to 576 rows is unsplit."""
    return 2 if 33 <= int(rows) <= 64 else 1


def dense_kernel_ksplit(rows, n, k):
    """The K split of the kernel AITER's dense group-32 MXFP8 GEMM on (16, 16)-preshuffled weights runs for
    ``rows x k -> n`` (32x32 weight scales): the ``_sk<N>`` of the tuned row's kernel name, the lookup the dense
    launch itself makes (aiter.ops.gemm_op_a8w8._gemm_mxscale_bpreshuffle). 0: no tuned row (the launch runs a
    heuristic kernel) or a name without a split. None: no AITER lookup in this process (stub tests).
    """
    try:
        from aiter.ops.gemm_op_a8w8 import get_mxscale_bpreshuffle_config

        row = get_mxscale_bpreshuffle_config(int(rows), int(n), int(k), "32x32")
    except Exception:  # noqa: BLE001
        # no AITER, or no device description to key the table with
        return None
    if row is None:
        return 0
    split = str(row.get("kernelName", "")).rpartition("_sk")[2].split("_")[0]
    return int(split) if split.isdigit() else 0


def dense_gate_up_ksplit(rows, k_dim=5120, inter_dim=2304):
    """The K split (1 or 2) the shared tile bodies must run GEMM1 with so that ``rows`` rows get the bits of the
    dense launches, or 0 when they cannot: the dense gate_up kernel of that row count is unsplit or split-K 2
    (``0 + p0 + p1``), the dense down kernel unsplit. Read from AITER's live tuned table, so a re-tuned row moves
    the split (or turns the fused path off for those rows) instead of changing the output; without an AITER lookup
    the table of d29c94315."""
    rows = int(rows)
    if rows <= 0:
        return 0
    gate_up = dense_kernel_ksplit(rows, 2 * inter_dim, k_dim)
    if gate_up is None:
        return _static_gate_up_ksplit(rows)
    down = dense_kernel_ksplit(rows, k_dim, inter_dim)
    return gate_up if gate_up in (1, 2) and down == 1 else 0


def tile_scale_byte(m_tile, n_tile, row, group, inter_dim):
    """Byte of the E8M0 scale of activation columns ``128 n_tile + 32 group ..+32`` of tile row ``row`` in the
    "tile" scale layout: [m-tile][128-column tile][32 rows][4 groups]. One GEMM1 workgroup writes the 128 bytes of
    its (m-tile, n-tile), i.e. whole cache lines of its own."""
    return m_tile * inter_dim + n_tile * 128 + row * 4 + group


def _byte_tensor(addr):
    """A byte tensor view of a raw device address (what ATileLoader takes as ``x_tensor``)."""
    ptr_ty = fx.PointerType.get(T.i8, fx.AddressSpace.Global, 1)
    return fx.Tensor(
        fx.make_view(fx.inttoptr(ptr_ty, fx.Int64(addr)), fx.make_layout(1 << 30, 1))
    )


class _NoExpert:
    """TileScheduler stand-in: one weight, so a tile's B row base is its column base."""

    def gate_base_row(self, expert_i32):
        return fx.Int32(0)


class MfmaScaleF8B(MfmaScaleGU):
    """MfmaScaleGU for FP8 A x FP8 B: the same call order, 8-dword B operands."""

    def __init__(self, *, m_repeat, num_acc_n):
        self._m_repeat = m_repeat
        self._m_active = m_repeat
        self._num_acc_n = num_acc_n
        self._is_fp4 = False
        self._atoms = {
            (osa, osb): fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(
                    16,
                    16,
                    128,
                    fx.Float8E4M3FN,
                    fx.Float8E4M3FN,
                    opsel_a=osa,
                    opsel_b=osb,
                )
            )
            for osa in range(4)
            for osb in range(4)
        }
        self.zero_value = fx.full(4, 0.0, fx.Float32)

    def _mfma(self, a_op, b_op, acc, sa_v, sb_v, ksub, ia, jb):
        a_frag = fx.make_rmem_tensor(8, fx.Int32)
        b_frag = fx.make_rmem_tensor(8, fx.Int32)
        c_frag = fx.make_rmem_tensor(4, fx.Float32)
        a_frag.store(Vec(a_op))
        b_frag.store(Vec(b_op))
        c_frag.store(Vec(acc))
        fx.gemm(
            self._atoms[(ksub * _PACK + ia, ksub * _PACK + jb)],
            c_frag,
            a_frag,
            b_frag,
            c_frag,
            scale_a=sa_v,
            scale_b=sb_v,
        )
        return c_frag.load()


class BWeightLoaderF8:
    """Per-K-step FP8 weights VMEM -> registers (i32x8 per 16 rows and 128 K), shuffle_weight(., (16, 16)) layout:
    16-byte chunk (16-row block n, 128-K step k, half p, K lane b, row r) at
    ``n * 16 * K + ((2 k + p) * 64 + 16 b + r) * 16`` holds W[16 n + r][128 k + 64 p + 16 b ..+16].

    ``gate_up_half`` = D > 0: the weight is ``[gate (D rows) | up (D rows)]`` and a wave's accumulator pairs are
    (gate, up) of the same 16 activation columns, as the routed GEMM1's interleaved weight gives them: accumulator
    ``ni`` reads rows ``row_base // 2 + 16 (ni // 2) + D (ni % 2)``. Otherwise it reads rows ``row_base + 16 ni``.
    """

    def __init__(self, *, w_rsrc, num_acc_n, k_dim, cache_modifier=0, gate_up_half=0):
        assert k_dim % 256 == 0
        self._w_rsrc = w_rsrc
        self._num_acc_n = num_acc_n
        self._k_dim = k_dim
        self._cache_modifier = int(cache_modifier)
        self._half = int(gate_up_half)
        self._lane = fx.thread_idx.x % 64

    def _load_pack(self, row_base_i32, ni, kstep_i32, ksub):
        row = gate_up_row(fx.Int32(row_base_i32), ni, self._half)
        k128 = fx.Int32(kstep_i32) * fx.Int32(2) + fx.Int32(ksub)
        halves = [
            Vec(
                buf_copy_load(
                    self._w_rsrc,
                    f8_weight_group(
                        row,
                        k128,
                        p,
                        fx.Int32(self._lane // 16),
                        fx.Int32(self._lane % 16),
                        self._k_dim,
                    ),
                    fx.Int32,
                    unit_elems=4,
                    cache_modifier=self._cache_modifier,
                )
            ).bitcast(fx.Int64)
            for p in range_constexpr(2)
        ]
        lo, hi = halves
        return Vec.from_elements([lo[0], lo[1], hi[0], hi[1]], fx.Int64).bitcast(
            fx.Int32
        )

    def load_step(self, row_base_i32, kstep_i32):
        return [
            self.load_ni(row_base_i32, ni, kstep_i32)
            for ni in range_constexpr(self._num_acc_n)
        ]

    def load_ni(self, row_base_i32, ni, kstep_i32):
        return [
            self._load_pack(row_base_i32, ni, kstep_i32, ks)
            for ks in range_constexpr(_PACK)
        ]


class BScaleLoaderRow32:
    """Row-major 32x32-block E8M0 weight scales [rows / 32, K / 32] as the packed words the scaled MFMA takes: one
    i32 per accumulator pair and K-step, byte ``2 ksub + (ni % 2)`` = the block scale of that accumulator's rows at
    K group ``lane // 16`` of 128-K step ``ksub`` (all 16 row lanes of a block share it).
    """

    def __init__(self, *, scale_rsrc, num_acc_n, k_dim, gate_up_half=0):
        assert num_acc_n % _PACK == 0 and (k_dim // 32) % 8 == 0
        self._rsrc = scale_rsrc
        self._n_pairs = num_acc_n // _PACK
        self._k_dim = k_dim
        self._half = int(gate_up_half)
        self._shift = (fx.thread_idx.x % 64) // 16 * 8

    def _bytes(self, row32_i32, kstep_i32, cache_modifier):
        """The lane's two scale bytes (128-K steps 0 and 1) of scale row ``row32`` at this K-step."""
        out = []
        for ks in range_constexpr(_PACK):
            off = scale32_dword(
                fx.Int32(row32_i32), fx.Int32(kstep_i32), ks, self._k_dim
            )
            if cache_modifier == 0:
                word = self._rsrc[off]
            else:
                word = buf_copy_load(
                    self._rsrc, off, fx.Int32, cache_modifier=cache_modifier
                )
            out.append((fx.Int32(word) >> fx.Int32(self._shift)) & fx.Int32(0xFF))
        return out

    def load_step(self, base_row_i32, kstep_i32, cache_modifier=0):
        if self._half:
            first = base_row_i32 // fx.Int32(
                64
            )  # the wave's 32 activation columns are one 32-row gate block
            gate = self._bytes(first, kstep_i32, cache_modifier)
            up = self._bytes(
                first + fx.Int32(self._half // 32), kstep_i32, cache_modifier
            )
            word = (
                gate[0]
                | (up[0] << fx.Int32(8))
                | (gate[1] << fx.Int32(16))
                | (up[1] << fx.Int32(24))
            )
            return [word for _ in range_constexpr(self._n_pairs)]
        out = []
        for pair in range_constexpr(self._n_pairs):
            s = self._bytes(
                base_row_i32 // fx.Int32(32) + fx.Int32(pair), kstep_i32, cache_modifier
            )
            out.append(
                s[0]
                | (s[0] << fx.Int32(8))
                | (s[1] << fx.Int32(16))
                | (s[1] << fx.Int32(24))
            )
        return out


class _KOffsetA:
    """ATileLoader shifted by whole K-steps (the second half of a split-K tile)."""

    def __init__(self, inner, k_bytes):
        self._inner, self._k_bytes = inner, k_bytes

    @property
    def _chunks(self):
        return self._inner._chunks

    def for_tile(self, *args, **kwargs):
        return self._inner.for_tile(*args, **kwargs)

    def load_regs(self, k_step_byte_off):
        return self._inner.load_regs(
            fx.Int32(k_step_byte_off) + fx.Int32(self._k_bytes)
        )

    def store(self, *args, **kwargs):
        return self._inner.store(*args, **kwargs)


class _KOffsetAScale:
    """AScaleLoader shifted by whole K-steps; the tile's scales are already staged."""

    def __init__(self, inner, k_steps):
        self._inner, self._k_steps = inner, k_steps

    def stage(self, *args, **kwargs):
        return None

    def load_step(self, lds_ascale, kstep_i32):
        return self._inner.load_step(
            lds_ascale, fx.Int32(kstep_i32) + fx.Int32(self._k_steps)
        )


class _KOffsetAScaleOwn(_KOffsetAScale):
    """_KOffsetAScale for a workgroup that runs only the second K half: it stages the tile's scales itself."""

    def stage(self, *args, **kwargs):
        return self._inner.stage(*args, **kwargs)


class _KOffsetB:
    """A B weight or B scale loader shifted by whole K-steps."""

    def __init__(self, inner, k_steps):
        self._inner, self._k_steps = inner, k_steps

    def _k(self, kstep_i32):
        return fx.Int32(kstep_i32) + fx.Int32(self._k_steps)

    def load_step(self, row_base_i32, kstep_i32, *args):
        return self._inner.load_step(row_base_i32, self._k(kstep_i32), *args)

    def load_ni(self, row_base_i32, ni, kstep_i32):
        return self._inner.load_ni(row_base_i32, ni, self._k(kstep_i32))


class AScaleTileLoader(AScaleLoader):
    """AScaleLoader over the "tile" scale layout (tile_scale_byte): stages a 32-row tile's scales to LDS in the
    row-major order ``load_step`` reads, one dword (four K groups) a thread and round.
    """

    def __init__(self, *, scale_addr, rows, k_dim, lds_bytes):
        assert k_dim % 128 == 0
        tiles = (rows + fx.Int32(SORT_BLOCK_M - 1)) // fx.Int32(SORT_BLOCK_M)
        super().__init__(
            scale_rsrc=ptr_buf_tensor(
                fx.Int64(scale_addr),
                fx.Int32,
                num_records_bytes=tiles * fx.Int32(k_dim),
            ),
            m_repeat=M_REPEAT,
            model_dim=k_dim,
            sort_block_m=SORT_BLOCK_M,
            total_threads=THREADS,
        )
        self._blocks = k_dim // 128  # dwords a row
        self._rounds = -(-SORT_BLOCK_M * self._blocks // THREADS)
        # The last round also copies dwords past the tile (rows the MFMAs never read).
        assert (
            self._rounds * THREADS * 4 <= lds_bytes
        ), "the A-scale LDS is too small for the tile-layout staging"

    def stage(self, lds_ascale, tile_row_base_i32, *args, **kwargs):
        base = fx.Int32(fx.ptrtoint(lds_ascale.ptr))
        tile_dword = (tile_row_base_i32 // fx.Int32(SORT_BLOCK_M)) * fx.Int32(
            SORT_BLOCK_M * self._blocks
        )
        for r in range_constexpr(self._rounds):
            d = fx.Int32(r * THREADS) + fx.Int32(self._tx)
            row = d // fx.Int32(self._blocks)
            block = d - row * fx.Int32(self._blocks)
            word = buf_copy_load(
                self._rsrc, tile_dword + block * fx.Int32(SORT_BLOCK_M) + row, fx.Int32
            )
            fx.ptr_store(
                fx.Int32(word), lds_typed_ptr(base + d * fx.Int32(4), T.i32, align=4)
            )


class _KeepPartial:
    """Epilogue stand-in of a split-K tile's first half: parks its accumulators in LDS (16 bytes each, one slot
    per thread), so the second half's loop has the registers of an unsplit tile."""

    def __init__(self, lds_partial):
        self._base = fx.recast_iter(fx.Float32, lds_partial.ptr)
        self.count = 0

    def _slot(self, index):
        return fx.add_offset(
            self._base,
            fx.make_int_tuple(
                fx.thread_idx.x * fx.Int32(self.count * 4) + fx.Int32(index * 4)
            ),
        )

    def store(self, acc, *args, **kwargs):
        self.count = len(acc)
        for index, a in enumerate(acc):
            fx.ptr_store(Vec(a), self._slot(index))

    def load(self, index):
        view = fx.make_view(
            fx.recast_iter(fx.Float32, self._slot(index)), fx.make_layout(4, 1)
        )
        return Vec(view.load())


class _SumPartials:
    """Epilogue of a split-K tile's last half: ``0 + p0 + p1`` (the dense split-K merge), then the real epilogue."""

    def __init__(self, first, epilogue):
        self._first, self._epilogue = first, epilogue

    def store(self, acc, *args, **kwargs):
        total = [
            (Vec.filled(4, 0.0, fx.Float32) + self._first.load(index)) + Vec(p1)
            for index, p1 in enumerate(acc)
        ]
        self._epilogue.store(total, *args, **kwargs)


def partial_slot(unit, thread, index):
    """16-byte slot, in the partials buffer, of accumulator ``index`` of thread ``thread`` of tile ``unit``: a
    thread's accumulators fill two cache lines of its own."""
    return (unit * THREADS + thread) * (M_REPEAT * NUM_ACC_N) + index


class _StorePartials:
    """Epilogue stand-in of a split-K tile's first half when another workgroup runs the second: the accumulators
    go to memory as raw FP32 (``partial_slot``), written through so that the other workgroup's XCD reads them.
    """

    def __init__(self, rsrc, unit, store_cm):
        self._rsrc, self._unit, self._store_cm = rsrc, unit, int(store_cm)

    def store(self, acc, *args, **kwargs):
        assert len(acc) == M_REPEAT * NUM_ACC_N
        for index, a in enumerate(acc):
            buf_copy_store(
                self._rsrc,
                partial_slot(self._unit, fx.thread_idx.x, index),
                Vec(a).bitcast(fx.Int32),
                fx.Int32,
                unit_elems=4,
                cache_modifier=self._store_cm,
            )


class _SumStoredPartials:
    """Epilogue of a split-K tile's second half on its own workgroup: ``wait()`` (the first half's partials are in
    memory), then ``0 + p0 + p1`` (the dense split-K merge, as _SumPartials) and the real epilogue.
    """

    def __init__(self, rsrc, unit, wait, epilogue):
        self._rsrc, self._unit, self._wait, self._epilogue = rsrc, unit, wait, epilogue

    def store(self, acc, *args, **kwargs):
        assert len(acc) == M_REPEAT * NUM_ACC_N
        self._wait()
        total = []
        for index, p1 in enumerate(acc):
            p0 = Vec(
                buf_copy_load(
                    self._rsrc,
                    partial_slot(self._unit, fx.thread_idx.x, index),
                    fx.Int32,
                    unit_elems=4,
                )
            )
            total.append(
                (Vec.filled(4, 0.0, fx.Float32) + p0.bitcast(fx.Float32)) + Vec(p1)
            )
        self._epilogue.store(total, *args, **kwargs)


def _set_fp32_denormals(mode):
    llvm_ops.call_intrinsic(
        None,
        "llvm.amdgcn.s.setreg",
        [fx.Int32(sq._HWREG_MODE_FP32_DENORM).ir_value(), fx.Int32(mode).ir_value()],
        [],
        [],
    )


def _pinned(words):
    """``words`` (an i32 x2 vector) through an empty inline asm with side effects: the same value, opaque to the
    optimizer, so the float operations that read it are issued after it. This keeps the activation's arithmetic
    behind the mode switch. It does not order ``v_cvt_pk_bf16_f32`` against the switch: this compiler schedules
    that instruction across ``s_setreg`` (see RefActQuantEpilogue)."""
    ty = ir.VectorType.get([2], ir.IntegerType.get_signless(32))
    return Vec(
        llvm_ops.inline_asm(ty, [_raw(words)], "", "=v,0", has_side_effects=True)
    )


class RefActQuantEpilogue:
    """GEMM1 epilogue with the dense path's numbers: gate / up rounded to BF16, then ``silu_mul_mxfp8`` (clamped
    SiLU-mul rounded to BF16, group-32 MXFP8 with E8M0 = ceil_pow2(amax / 448)) through AITER's transcription of
    that kernel, in its float mode (FP32 denormals flushed; restored at the end). Writes E4M3 ``[rows, inter]`` and
    row-major E8M0 ``[rows, inter / 32]``; rows past the tile's valid count are not stored.

    Float mode: the compiler may issue some of the accumulators' BF16 conversions after the switch (flushed mode)
    instead of before it. That only matters for an FP32-denormal accumulator, which the flushed conversion turns
    into a signed zero where the dense GEMM stores a BF16 denormal; ``silu_mul_mxfp8`` reads a denormal gate or up
    as that same signed zero (its own mode), so the pair it writes is the same either way.

    Accumulators ``2 p`` / ``2 p + 1`` of a 16-row group are gate / up of activation columns ``16 p ..`` of the
    wave's 32 (BWeightLoaderF8 gate_up_half). The products go through LDS as the routed epilogue's do, so a lane of
    the store pass owns eight consecutive columns of a row and four adjacent lanes one 32-column group (a DPP quad).
    """

    def __init__(
        self,
        *,
        out_addr,
        out_scale_addr,
        rows,
        inter_dim,
        limit,
        lds_out,
        store_cm=0,
        scale_layout="row",
    ):
        assert (
            inter_dim % 128 == 0
            and float(limit) > 0.0
            and scale_layout in ("row", "tile")
        )
        self._inter = inter_dim
        self._limit = float(limit)
        self._lds_out = lds_out
        self._store_cm = int(store_cm)
        # "tile": tile_scale_byte's layout (AScaleTileLoader reads it), whose bytes no two workgroups share a
        # cache line of; "row": row-major [rows, inter / 32], what the dense down GEMM reads.
        self._tile_scales = scale_layout == "tile"
        self._out = ptr_buf_tensor(
            fx.Int64(out_addr),
            fx.Int32,
            unit_elems=2,
            num_records_bytes=rows * fx.Int32(inter_dim),
        )
        scale_bytes = rows * fx.Int32(inter_dim // 32)
        if self._tile_scales:
            scale_bytes = (
                (rows + fx.Int32(SORT_BLOCK_M - 1)) // fx.Int32(SORT_BLOCK_M)
            ) * fx.Int32(inter_dim)
        self._out_scale = ptr_buf_tensor(
            fx.Int64(out_scale_addr), fx.Int8, num_records_bytes=scale_bytes
        )

    def store(
        self,
        acc,
        tile_i32,
        tile_row_base_i32,
        n_tile_base_i32,
        valid_rows_i32=None,
        store_cm=None,
    ):
        tx = fx.thread_idx.x
        wave = tx // fx.Int32(64)
        lane = tx % fx.Int32(64)
        l16 = lane % fx.Int32(16)
        ld4 = (lane // fx.Int32(16)) * fx.Int32(4)
        cs_tile_n = TILE_N // 2
        cbase = wave * fx.Int32(cs_tile_n // NUM_WAVES)
        cptr = self._lds_out.ptr
        pairs = NUM_ACC_N // _PACK

        # The dense GEMM's BF16 rounding, in the kernel's own float mode.
        rounded = [
            _pinned(Vec(acc[mi * NUM_ACC_N + ni]).to(fx.BFloat16).bitcast(fx.Int32))
            for mi in range_constexpr(M_REPEAT)
            for ni in range_constexpr(NUM_ACC_N)
        ]
        _set_fp32_denormals(0)
        # Everything from here to the stores reads the BF16 words through a second pin, behind the switch.
        rounded = [_pinned(words) for words in rounded]
        gate = [
            [rounded[i][0], rounded[i][1]]
            for i in range_constexpr(0, len(rounded), _PACK)
        ]
        up = [
            [rounded[i][0], rounded[i][1]]
            for i in range_constexpr(1, len(rounded), _PACK)
        ]
        for mi in range_constexpr(M_REPEAT):
            gate_words = [
                w for p in range_constexpr(pairs) for w in gate[mi * pairs + p]
            ]
            up_words = [w for p in range_constexpr(pairs) for w in up[mi * pairs + p]]
            act = sq.silu_mul_bf16(gate_words, up_words, self._limit)
            for w in range_constexpr(len(act)):
                vals = (
                    Vec.from_elements([act[w]], fx.Int32)
                    .bitcast(fx.BFloat16)
                    .to(fx.Float32)
                )
                col = cbase + fx.Int32((w // 2) * 16) + l16
                for e in range_constexpr(2):
                    row = fx.Int32(mi * 16 + (w % 2) * 2 + e) + ld4
                    ptr = fx.add_offset(
                        cptr, fx.make_int_tuple(row * fx.Int32(cs_tile_n) + col)
                    )
                    fx.ptr_store(Vec.from_elements([vals[e]], fx.Float32), ptr)
        gpu.barrier()

        lanes_per_row = cs_tile_n // sq.LANE_VALUES
        rows_per_iter = THREADS // lanes_per_row
        mlane = tx // fx.Int32(lanes_per_row)
        nlane = tx % fx.Int32(lanes_per_row)
        col0 = nlane * fx.Int32(sq.LANE_VALUES)
        gcol = n_tile_base_i32 // fx.Int32(2) - cbase + col0
        is_writer = (gcol & fx.Int32(31)) == fx.Int32(0)
        for mr in range_constexpr(SORT_BLOCK_M // rows_per_iter):
            row = fx.Int32(mr * rows_per_iter) + mlane
            row_g = tile_i32 * fx.Int32(SORT_BLOCK_M) + row
            f32_iter = fx.recast_iter(
                fx.Float32,
                fx.add_offset(
                    cptr, fx.make_int_tuple(row * fx.Int32(cs_tile_n) + col0)
                ),
            )
            frag = fx.make_view(f32_iter, fx.make_layout(sq.LANE_VALUES, 1)).load()
            words = [
                Vec.from_elements([frag[2 * w], frag[2 * w + 1]], fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Int32)[0]
                for w in range_constexpr(sq.LANE_VALUES // 2)
            ]
            code = sq.group_code(words)
            quant = sq.quant_words(words, code)
            valid = fx.Boolean(True) if valid_rows_i32 is None else row < valid_rows_i32
            out_byte = valid.select(
                row_g * fx.Int32(self._inter) + gcol, fx.Int32(_OOB)
            )
            buf_copy_store(
                self._out,
                out_byte // fx.Int32(8),
                Vec.from_elements(quant, fx.Int32),
                fx.Int32,
                unit_elems=2,
                cache_modifier=self._store_cm,
            )
            if self._tile_scales:
                scale_at = tile_scale_byte(
                    tile_i32,
                    gcol >> fx.Int32(7),
                    row,
                    (gcol >> fx.Int32(5)) & fx.Int32(3),
                    self._inter,
                )
            else:
                scale_at = row_g * fx.Int32(self._inter // 32) + (gcol >> fx.Int32(5))
            scale_byte = (valid & is_writer).select(scale_at, fx.Int32(_OOB))
            buf_copy_store(
                self._out_scale,
                scale_byte,
                fx.Int8(code),
                fx.Int8,
                cache_modifier=self._store_cm,
            )
        wait_lds_barrier()
        _set_fp32_denormals(_FP32_DENORM_KEPT)


class Bf16RowsEpilogue:
    """GEMM2 epilogue of the shared expert: the tile's accumulators rounded to BF16 into one row-major
    ``[rows, n_out]`` tensor (no route weight, no scatter). Rows past the tile's valid count are not stored.
    """

    def __init__(self, *, out_addr, rows, n_out, lds_out, store_cm=0):
        assert n_out % TILE_N == 0
        self._n_out = n_out
        self._lds_out = lds_out
        self._store_cm = int(store_cm)
        self._out = ptr_buf_tensor(
            fx.Int64(out_addr),
            fx.Int32,
            unit_elems=4,
            num_records_bytes=rows * fx.Int32(n_out * 2),
        )

    def store(
        self,
        acc,
        tile_i32,
        tile_row_base_i32,
        n_tile_base_i32,
        valid_rows_i32=None,
        store_cm=None,
    ):
        tx = fx.thread_idx.x
        wave = tx // fx.Int32(64)
        lane = tx % fx.Int32(64)
        l16 = lane % fx.Int32(16)
        ld4 = (lane // fx.Int32(16)) * fx.Int32(4)
        wave_base = wave * fx.Int32(TILE_N // NUM_WAVES)
        cptr = self._lds_out.ptr
        for mi in range_constexpr(M_REPEAT):
            for ni in range_constexpr(NUM_ACC_N):
                v4 = Vec(acc[mi * NUM_ACC_N + ni])
                col = wave_base + fx.Int32(ni * 16) + l16
                for ii in range_constexpr(4):
                    row = fx.Int32(mi * 16 + ii) + ld4
                    ptr = fx.add_offset(
                        cptr, fx.make_int_tuple(row * fx.Int32(TILE_N) + col)
                    )
                    fx.ptr_store(Vec.from_elements([v4[ii]], fx.Float32), ptr)
        gpu.barrier()

        lanes_per_row = TILE_N // 8
        rows_per_iter = THREADS // lanes_per_row
        mlane = tx // fx.Int32(lanes_per_row)
        nlane = tx % fx.Int32(lanes_per_row)
        col0 = nlane * fx.Int32(8)
        gcol = n_tile_base_i32 - wave_base + col0
        for mr in range_constexpr(SORT_BLOCK_M // rows_per_iter):
            row = fx.Int32(mr * rows_per_iter) + mlane
            row_g = tile_i32 * fx.Int32(SORT_BLOCK_M) + row
            f32_iter = fx.recast_iter(
                fx.Float32,
                fx.add_offset(cptr, fx.make_int_tuple(row * fx.Int32(TILE_N) + col0)),
            )
            frag = fx.make_view(f32_iter, fx.make_layout(8, 1)).load()
            res = Vec.from_elements(
                [frag[i] for i in range_constexpr(8)], fx.Float32
            ).to(fx.BFloat16)
            valid = fx.Boolean(True) if valid_rows_i32 is None else row < valid_rows_i32
            out_byte = valid.select(
                (row_g * fx.Int32(self._n_out) + gcol) * fx.Int32(2), fx.Int32(_OOB)
            )
            buf_copy_store(
                self._out,
                out_byte // fx.Int32(16),
                res.bitcast(fx.Int32),
                fx.Int32,
                unit_elems=4,
                cache_modifier=self._store_cm,
            )
        wait_lds_barrier()


def build_shared_tile(
    *,
    x_addr,
    scale_x_addr,
    w_addr,
    scale_w_addr,
    rows,
    k_dim,
    epilogue,
    a_buf,
    a_scale_lds,
    gate_up_half=0,
    ksplit=1,
    b_cache_modifier=0,
    partial_lds=None,
    a_scale_layout="row",
    a_scale_lds_bytes=0,
):
    """``run(m_tile, n_tile)``: one 32-row x 256-column tile of ``x [rows, k_dim] @ w.T`` through ``epilogue``.

    ``x_addr`` / ``scale_x_addr``: E4M3 rows and their row-major 1x32 E8M0 scales (readable up to the next 16-byte
    boundary past ``rows * k_dim / 32``); ``w_addr`` / ``scale_w_addr``: the (16, 16)-preshuffled E4M3 weight and
    its row-major 32x32 E8M0 scales. ``gate_up_half``: BWeightLoaderF8's. ``ksplit``: 1, or 2 for the split-K 2
    association (``partial_lds``: 16 bytes per accumulator and thread of LDS for the first half's partials).
    ``a_scale_layout="tile"``: the A scales are in tile_scale_byte's layout (``a_scale_lds_bytes``: the size of
    ``a_scale_lds``), as RefActQuantEpilogue(scale_layout="tile") writes them.
    """
    k_iters = k_dim // K_STEP
    assert (
        k_dim % K_STEP == 0
        and ksplit in (1, 2)
        and k_iters % ksplit == 0
        and k_iters // ksplit >= 2
    )
    n_scale = k_dim // 32
    scale_bytes = (rows * fx.Int32(n_scale) + fx.Int32(15)) & fx.Int32(-16)
    a_gather = ATileLoader(
        row_bytes=k_dim,
        sort_block_m=SORT_BLOCK_M,
        k_step_bytes=K_STEP,
        total_threads=THREADS,
        swizzle=True,
        x_tensor=_byte_tensor(x_addr),
        async_copy=False,
        a_dtype="fp8",
    )
    a_s2r = AS2RLoader(k_step_bytes=K_STEP, swizzle=True, a_dtype="fp8")
    if a_scale_layout == "tile":
        a_scale = AScaleTileLoader(
            scale_addr=scale_x_addr, rows=rows, k_dim=k_dim, lds_bytes=a_scale_lds_bytes
        )
    else:
        a_scale = AScaleLoader(
            scale_rsrc=ptr_buf_tensor(
                fx.Int64(scale_x_addr),
                fx.Int32,
                unit_elems=4,
                num_records_bytes=scale_bytes,
            ),
            m_repeat=M_REPEAT,
            model_dim=k_dim,
            sort_block_m=SORT_BLOCK_M,
            total_threads=THREADS,
        )
    b_loader = BWeightLoaderF8(
        w_rsrc=ptr_buf_tensor(fx.Int64(w_addr), fx.Int32, unit_elems=4),
        num_acc_n=NUM_ACC_N,
        k_dim=k_dim,
        cache_modifier=b_cache_modifier,
        gate_up_half=gate_up_half,
    )
    b_scale = BScaleLoaderRow32(
        scale_rsrc=ptr_buf_tensor(fx.Int64(scale_w_addr), fx.Int32),
        num_acc_n=NUM_ACC_N,
        k_dim=k_dim,
        gate_up_half=gate_up_half,
    )
    mfma = MfmaScaleF8B(m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N)
    sched = _NoExpert()
    a_lds_i32 = SORT_BLOCK_M * K_STEP // 4
    wave_base = (fx.thread_idx.x // 64) * fx.Int32(TILE_N // NUM_WAVES)

    def tile(m_tile, n_tile_base, meta, a_g, a_sc, b_l, b_sc, epi, steps):
        # fmt: off
        do_tile_la(m_tile, n_tile_base, fx.Int32(0), sched, a_g, a_s2r, b_l, b_sc, a_sc, mfma, epi, a_buf,
            a_scale_lds, a_lds_i32, steps, M_REPEAT, NUM_ACC_N, K_STEP, None, None, 1, None, meta, 1)
        # fmt: on

    def run(m_tile, n_tile):
        n_tile_base = wave_base + n_tile * fx.Int32(TILE_N)
        tile_row_base = m_tile * fx.Int32(SORT_BLOCK_M)
        left = rows - tile_row_base
        meta = (
            tile_row_base,
            (left < fx.Int32(SORT_BLOCK_M)).select(left, fx.Int32(SORT_BLOCK_M)),
        )
        if ksplit == 1:
            tile(
                m_tile,
                n_tile_base,
                meta,
                a_gather,
                a_scale,
                b_loader,
                b_scale,
                epilogue,
                k_iters,
            )
            return
        half = k_iters // 2
        first = _KeepPartial(partial_lds)
        tile(
            m_tile, n_tile_base, meta, a_gather, a_scale, b_loader, b_scale, first, half
        )
        tile(
            m_tile,
            n_tile_base,
            meta,
            _KOffsetA(a_gather, half * K_STEP),
            _KOffsetAScale(a_scale, half),
            _KOffsetB(b_loader, half),
            _KOffsetB(b_scale, half),
            _SumPartials(first, epilogue),
            half,
        )

    return run


# fmt: off
def build_shared_half_tiles(*, x_addr, scale_x_addr, w_addr, scale_w_addr, rows, k_dim, epilogue, a_buf, a_scale_lds,
    part_addr, n_tiles, units, wait, gate_up_half=0, b_cache_modifier=0, store_cm=0):
# fmt: on
    """``(first(m_tile, n_tile), second(m_tile, n_tile))``: the two K halves of build_shared_tile(ksplit=2)'s tile
    as two runners for two workgroups, with the same numbers. ``first`` runs K steps [0, K / 2) from zero and
    leaves its accumulators in memory at ``part_addr`` (PARTIAL_UNIT_BYTES per tile, tile = m_tile * ``n_tiles`` +
    n_tile, ``units`` tiles in all; stores with ``store_cm``); its caller then tells the other workgroup. ``second`` runs
    K steps [K / 2, K) from zero, calls ``wait()`` (returns once the first half's partials are readable), sums
    ``0 + p0 + p1`` and runs ``epilogue``. Other arguments as build_shared_tile."""
    k_iters = k_dim // K_STEP
    half = k_iters // 2
    assert k_dim % K_STEP == 0 and k_iters % 2 == 0 and half >= 2
    n_scale = k_dim // 32
    scale_bytes = (rows * fx.Int32(n_scale) + fx.Int32(15)) & fx.Int32(-16)
    a_gather = ATileLoader(row_bytes=k_dim, sort_block_m=SORT_BLOCK_M, k_step_bytes=K_STEP, total_threads=THREADS,
        swizzle=True, x_tensor=_byte_tensor(x_addr), async_copy=False, a_dtype="fp8")
    a_s2r = AS2RLoader(k_step_bytes=K_STEP, swizzle=True, a_dtype="fp8")
    a_scale = AScaleLoader(
        scale_rsrc=ptr_buf_tensor(fx.Int64(scale_x_addr), fx.Int32, unit_elems=4, num_records_bytes=scale_bytes),
        m_repeat=M_REPEAT, model_dim=k_dim, sort_block_m=SORT_BLOCK_M, total_threads=THREADS)
    b_loader = BWeightLoaderF8(w_rsrc=ptr_buf_tensor(fx.Int64(w_addr), fx.Int32, unit_elems=4), num_acc_n=NUM_ACC_N,
        k_dim=k_dim, cache_modifier=b_cache_modifier, gate_up_half=gate_up_half)
    b_scale = BScaleLoaderRow32(scale_rsrc=ptr_buf_tensor(fx.Int64(scale_w_addr), fx.Int32), num_acc_n=NUM_ACC_N,
        k_dim=k_dim, gate_up_half=gate_up_half)
    part = ptr_buf_tensor(fx.Int64(part_addr), fx.Int32, unit_elems=4,
        num_records_bytes=units * fx.Int32(PARTIAL_UNIT_BYTES))
    mfma = MfmaScaleF8B(m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N)
    sched = _NoExpert()
    a_lds_i32 = SORT_BLOCK_M * K_STEP // 4
    wave_base = (fx.thread_idx.x // 64) * fx.Int32(TILE_N // NUM_WAVES)

    def tile(m_tile, n_tile, a_g, a_sc, b_l, b_sc, epi):
        n_tile_base = wave_base + n_tile * fx.Int32(TILE_N)
        tile_row_base = m_tile * fx.Int32(SORT_BLOCK_M)
        left = rows - tile_row_base
        meta = (tile_row_base, (left < fx.Int32(SORT_BLOCK_M)).select(left, fx.Int32(SORT_BLOCK_M)))
        # fmt: off
        do_tile_la(m_tile, n_tile_base, fx.Int32(0), sched, a_g, a_s2r, b_l, b_sc, a_sc, mfma, epi, a_buf,
            a_scale_lds, a_lds_i32, half, M_REPEAT, NUM_ACC_N, K_STEP, None, None, 1, None, meta, 1)
        # fmt: on

    def first(m_tile, n_tile):
        unit = m_tile * fx.Int32(n_tiles) + n_tile
        tile(m_tile, n_tile, a_gather, a_scale, b_loader, b_scale, _StorePartials(part, unit, store_cm))

    def second(m_tile, n_tile):
        unit = m_tile * fx.Int32(n_tiles) + n_tile
        tile(m_tile, n_tile, _KOffsetA(a_gather, half * K_STEP), _KOffsetAScaleOwn(a_scale, half),
            _KOffsetB(b_loader, half), _KOffsetB(b_scale, half), _SumStoredPartials(part, unit, wait, epilogue))

    return first, second


def _lds_struct(k_dim, c_tile_bytes, ksplit=1):
    pool_bytes = max(2 * SORT_BLOCK_M * K_STEP, c_tile_bytes)

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, SORT_BLOCK_M * (k_dim // 32), 16]
        if ksplit > 1:
            partial: fx.Array[fx.Int8, THREADS * M_REPEAT * NUM_ACC_N * 16, 16]

    return SharedStorage


def _launcher(kernel, waves_per_eu_hint):
    @flyc.jit
    def launch(a0: fx.Int64, a1: fx.Int64, a2: fx.Int64, a3: fx.Int64, a4: fx.Int64, a5: fx.Int64, rows: fx.Int32,
        grid_x: fx.Int32, stream: fx.Stream):
        kernel(
            a0, a1, a2, a3, a4, a5, rows, grid_x,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": f"{THREADS},{THREADS}",
            },
        ).launch(grid=(fx.Int64(grid_x), 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


def gemm1_name(k_dim, inter_dim, limit, ksplit, b_cache_modifier):
    return (f"megamoe_shared_gemm1_t{SORT_BLOCK_M}x{TILE_N}x{K_STEP}_k{k_dim}_d{inter_dim}"
        f"_sl{str(float(limit)).replace('.', 'p')}_sk{ksplit}_bnt{b_cache_modifier}")


def gemm2_name(k_dim, n_out, b_cache_modifier):
    return f"megamoe_shared_gemm2_t{SORT_BLOCK_M}x{TILE_N}x{K_STEP}_k{k_dim}_n{n_out}_sk1_bnt{b_cache_modifier}"


@functools.cache
def compile_shared_gemm1(*, k_dim: int = 5120, inter_dim: int = 2304, limit: float = 10.0, ksplit: int = 1,
    b_cache_modifier: int = 0, waves_per_eu_hint: int = 2):
    """Standalone shared-expert GEMM1 + activation: one workgroup per (32-row tile, 128 activation columns)."""
    assert (2 * inter_dim) % TILE_N == 0
    n_tiles = (2 * inter_dim) // TILE_N
    storage = _lds_struct(k_dim, SORT_BLOCK_M * (TILE_N // 2) * 4, ksplit)

    @flyc.kernel(name=gemm1_name(k_dim, inter_dim, limit, ksplit, b_cache_modifier),
        known_block_size=[THREADS, 1, 1])
    def kernel(addr_out: fx.Int64, addr_out_scale: fx.Int64, addr_x: fx.Int64, addr_scale_x: fx.Int64,
        addr_w: fx.Int64, addr_scale_w: fx.Int64, rows: fx.Int32, grid_x: fx.Int32):
        lds = fx.SharedAllocator().allocate(storage).peek()
        epilogue = RefActQuantEpilogue(out_addr=addr_out, out_scale_addr=addr_out_scale, rows=rows,
            inter_dim=inter_dim, limit=limit, lds_out=_LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr)))
        run_tile = build_shared_tile(x_addr=addr_x, scale_x_addr=addr_scale_x, w_addr=addr_w,
            scale_w_addr=addr_scale_w, rows=rows, k_dim=k_dim, epilogue=epilogue, a_buf=lds.pool,
            a_scale_lds=lds.A_scale, gate_up_half=inter_dim, ksplit=ksplit, b_cache_modifier=b_cache_modifier,
            partial_lds=lds.partial if ksplit > 1 else None)
        total_work = ((rows + fx.Int32(SORT_BLOCK_M - 1)) // fx.Int32(SORT_BLOCK_M)) * fx.Int32(n_tiles)
        for flat in range(fx.block_idx.x, total_work, grid_x):
            m_tile = flat // fx.Int32(n_tiles)
            run_tile(m_tile, flat - m_tile * fx.Int32(n_tiles))

    return _launcher(kernel, waves_per_eu_hint)


@functools.cache
def compile_shared_gemm2(*, k_dim: int = 2304, n_out: int = 5120, b_cache_modifier: int = 0,
    waves_per_eu_hint: int = 2):
    """Standalone shared-expert GEMM2: one workgroup per (32-row tile, 256 output columns)."""
    assert n_out % TILE_N == 0
    n_tiles = n_out // TILE_N
    storage = _lds_struct(k_dim, SORT_BLOCK_M * TILE_N * 4)

    @flyc.kernel(name=gemm2_name(k_dim, n_out, b_cache_modifier), known_block_size=[THREADS, 1, 1])
    def kernel(addr_out: fx.Int64, addr_unused: fx.Int64, addr_x: fx.Int64, addr_scale_x: fx.Int64,
        addr_w: fx.Int64, addr_scale_w: fx.Int64, rows: fx.Int32, grid_x: fx.Int32):
        lds = fx.SharedAllocator().allocate(storage).peek()
        epilogue = Bf16RowsEpilogue(out_addr=addr_out, rows=rows, n_out=n_out,
            lds_out=_LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr)))
        run_tile = build_shared_tile(x_addr=addr_x, scale_x_addr=addr_scale_x, w_addr=addr_w,
            scale_w_addr=addr_scale_w, rows=rows, k_dim=k_dim, epilogue=epilogue, a_buf=lds.pool,
            a_scale_lds=lds.A_scale, b_cache_modifier=b_cache_modifier)
        total_work = ((rows + fx.Int32(SORT_BLOCK_M - 1)) // fx.Int32(SORT_BLOCK_M)) * fx.Int32(n_tiles)
        for flat in range(fx.block_idx.x, total_work, grid_x):
            m_tile = flat // fx.Int32(n_tiles)
            run_tile(m_tile, flat - m_tile * fx.Int32(n_tiles))

    return _launcher(kernel, waves_per_eu_hint)


def _grid(rows, n_tiles):
    return max(1, (int(rows) + SORT_BLOCK_M - 1) // SORT_BLOCK_M) * n_tiles


# fmt: off
def shared_gemm1(out_ptr, out_scale_ptr, x_ptr, scale_x_ptr, w_ptr, scale_w_ptr, rows, stream, *, k_dim=5120,
    inter_dim=2304, limit=10.0, ksplit=1, b_cache_modifier=0):
# fmt: on
    """Launch the standalone GEMM1 + activation on raw device addresses (see the module docstring for the formats):
    ``out`` E4M3 [rows, inter_dim], ``out_scale`` E8M0 [rows, inter_dim / 32]."""
    launch = compile_shared_gemm1(k_dim=k_dim, inter_dim=inter_dim, limit=float(limit), ksplit=ksplit,
        b_cache_modifier=b_cache_modifier)
    _run_compiled(launch, fx.Int64(out_ptr), fx.Int64(out_scale_ptr), fx.Int64(x_ptr), fx.Int64(scale_x_ptr),
        fx.Int64(w_ptr), fx.Int64(scale_w_ptr), fx.Int32(int(rows)),
        fx.Int32(_grid(rows, (2 * inter_dim) // TILE_N)), stream)


def shared_gemm2(out_ptr, x_ptr, scale_x_ptr, w_ptr, scale_w_ptr, rows, stream, *, k_dim=2304, n_out=5120,
    b_cache_modifier=0):
    """Launch the standalone GEMM2 on raw device addresses: ``out`` BF16 [rows, n_out]."""
    launch = compile_shared_gemm2(k_dim=k_dim, n_out=n_out, b_cache_modifier=b_cache_modifier)
    _run_compiled(launch, fx.Int64(out_ptr), fx.Int64(0), fx.Int64(x_ptr), fx.Int64(scale_x_ptr), fx.Int64(w_ptr),
        fx.Int64(scale_w_ptr), fx.Int32(int(rows)), fx.Int32(_grid(rows, n_out // TILE_N)), stream)
