# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Ported from amd-kernels ``gemm/mxfp8_common/mxfp8_gemm_4wave.py`` (the persistent 4-wave
# kernel the MiniMax-M3.1 prefill runs for N 6144 x K 8192 and the other large-M MXFP8 shapes),
# itself built on FlyDSL's ``fp4_gemm_4wave``; imports adapted to aiter's own buffer helpers.
# ruff: noqa: B023
# (B023: the ``extra=`` closures over loop variables are called inside the same iteration's
# ``_one_step``, at trace time, so no late-binding value can reach them.)
"""4-wave MXFP8 GEMM for gfx950: C[M, N] = A[M, K] @ W[N, K]^T with per-32 e8m0 scales on
both operands (the scaled MFMA applies them), bf16 out.

The structure of FlyDSL's ``fp4_gemm_4wave`` (HipKittens' 4-wave layout): a workgroup of
4 waves computes a 256x256 tile, each wave a 2x2 of 64x64 quadrants (c00 / c01 / c10 /
c11) as 4x4 tiles of ``v_mfma_scale_f32_16x16x128_f8f6f4``; A and W go through 8 LDS
buffers (cur / next x two 128-row halves x A / W, 16 KB each) filled by
``buffer_load ... lds`` two K-steps ahead, the LDS reads and the global loads of the next
step interleaved between the MFMAs of this one. Compared with the 8-wave (2x4) staged
kernel a wave reads half the LDS bytes per MFMA (A 16 KB + W 16 KB for 64 MFMAs).

MXFP8 specifics: a K-step is 128 K = 128 bytes per row (one MFMA K-block), which is half
a 256-K scale block; the scales of a block (AITER's shuffled layout, one dword per lane =
2 K-halves x 2 row-halves of a 32-row group) are gathered into LDS by one
``buffer_load_dwordx4 ... lds`` per wave per block, read back into VGPRs one step ahead,
and the MFMA picks its byte with op_sel = (tile parity in the 32-row / 32-column group)
and op_sel_hi = (K-step parity). A: row-major fp8 [M, K]; W: ``shuffle_weight(.., (16, 16))``;
scales: ``shuffle_scale_a16w4(.., 1, False)`` (the branch's ``preshuffle_mxfp8_weight`` /
``mxfp8_e4m3_quantize_flydsl`` outputs). N must be a multiple of 256; M is ragged (rows past M
read zero through the buffer bounds and their stores are dropped).

Persistent: min(tiles, 256) workgroups, each walking the tiles w, w + grid, ...; the next
tile's first loads and scale gather go out in the last two steps of the current tile, and
the stores of a finished quadrant are interleaved with the MFMAs that follow (the old and
the new accumulators share the 256 AGPRs half and half). The bf16 output goes through a
per-wave LDS transpose so every store writes whole 128-byte rows (``StoreCLds``).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops as _buffer_ops

_N_WAVES = 4
PARENT_M = 256  # the tile of the 4x4 layout; the 4x2 / 2x2 layouts run sub-tiles of it
PARENT_N = 256
BLOCK_K = 128  # one MFMA K-block per step; a scale block is two steps
_PACK = 2  # 16-row tiles per 32-row scale group


def row_stationary_order(n_a: int, n_b: int) -> list[tuple[int, int]]:
    """(A tile, B tile) issue order of one quadrant, row by row: A tile i stays for ``n_b``
    consecutive MFMAs while the B tiles serpentine, so adjacent MFMAs switch one operand.
    Every accumulator still takes its K-steps in order, so the output is bitwise the zig-zag
    order's."""
    cols = list(range(n_b))
    return [(i, j) for i in range(n_a) for j in (cols[::-1] if i % 2 else cols)]


_LDS_BUF_BYTES = 16384


def _ceildiv(a, b):
    return (a + b - 1) // b


def _swizzle_128(row, col):
    """XOR bank swizzle of a 128-byte LDS row (16-byte chunks)."""
    offset = row * 128 + col
    swizzle = ((offset % (16 * 128)) >> 8) << 4
    swizzled = offset ^ swizzle
    return swizzled // 128, swizzled % 128


def _global_swizzle(lane_id, wave_id, K, n_rounds, preshuffled):
    """Per-lane global byte offsets of the 16-byte pieces a wave's g2s loads move (the
    LDS destination is the linear m0 stream, so the swizzle sits on the global side)."""
    offsets = []
    for rnd in range_constexpr(n_rounds):
        if const_expr(preshuffled):
            row = lane_id % 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
            col = (lane_id // 8) * 16
            offsets.append(
                (row // 16) * (K * 16)
                + (row % 16) * 16
                + (col // 64) * 1024
                + ((col % 64) // 16) * 256
                + (col % 16)
            )
        else:
            row = lane_id // 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
            col = (lane_id % 8) * 16
            r, c = _swizzle_128(row, col)
            offsets.append(r * K + c)
    return offsets


class _Buf:
    def __init__(self, base_ptr, byte_off):
        self.base_ptr = base_ptr
        self.byte_off = byte_off


_gep = _buffer_ops.get_element_ptr


def _lds_ptr_t():
    # Raw LDS pointer type (D10b of the PR description): kept instead of the backend-resolved
    # ``fx.to_llvm_ptr(fx.inttoptr(PointerType(.., Shared), addr))`` so that the instruction
    # stream measured and validated for this kernel does not change under this port.
    return _ir.Type.parse("!llvm.ptr<3>")


def _asm_void(operands, asm_string, constraints, clobbers=""):
    if clobbers:
        constraints = f"{constraints},{clobbers}"
    _llvm.inline_asm(None, operands, asm_string, constraints, has_side_effects=True)


def _cvt_pk_bf16(a, b):
    v2f32 = _ir.VectorType.get([2], fx.Float32.ir_type)
    vec = Vec.from_elements([fx.Float32(a), fx.Float32(b)], fx.Float32)
    src = fx.as_ir_value(vec)
    if src.type != v2f32:
        src = fx.arith.bitcast(v2f32, src)
    v2bf16 = _ir.VectorType.get([2], fx.BFloat16.ir_type)
    return _llvm.BitcastOp(fx.Int32.ir_type, fx.arith.trunc_f(v2bf16, src)).result


_M0_CLOBBER = "~{m0}"


def wait_barrier(count):
    """``s_waitcnt vmcnt(count) lgkmcnt(0)`` + ``s_barrier``"""
    _rocdl.s_waitcnt(vmcnt=count, lgkmcnt=0)
    _rocdl.s_barrier()


def _uniform_i32(value):
    raw = fx.as_ir_value(value) if not isinstance(value, _ir.Value) else value
    if raw.type != _T.i32:
        raw = fx.as_ir_value(fx.Int32(raw))
    return _rocdl.readfirstlane(_T.i32, raw)


class G2SLoaderAsm:
    """Global -> LDS through ``buffer_load_dwordx4 ... lds`` (m0 = the LDS destination):
    one instruction per wave moves 1 KB (64 lanes x 16 B); a buffer of 128 rows x 128 B
    takes 16 of them = 4 per wave x 4 rounds."""

    def __init__(self, rsrc, gl_offsets, n_load_steps, wave_id):
        self.rsrc = fx.as_ir_value(rsrc)
        self.gl_offsets = gl_offsets
        self.n_load_steps = n_load_steps
        self.wave_id = wave_id

    @property
    def _step_stride(self):
        return _N_WAVES * 1024

    def set_wave_base(self, base_ptr):
        wb = fx.Int32(fx.ptrtoint(base_ptr)) + fx.Int32(self.wave_id * 1024)
        self._wave_base_s = _rocdl.readfirstlane(_T.i32, fx.as_ir_value(wb))

    def _emit(self, lds_dst, k_offset, step):
        voff = fx.as_ir_value(fx.Int32(self.gl_offsets[step]))
        soff = _uniform_i32(k_offset)
        stride = self._step_stride
        if step == 0:
            m0 = fx.as_ir_value(
                fx.Int32(self._wave_base_s) + fx.Int32(lds_dst.byte_off)
            )
            asm = "s_mov_b32 m0, $0\nbuffer_load_dwordx4 $1, $2, $3 offen lds"
            _asm_void([m0, voff, self.rsrc, soff], asm, "s,v,s,s", _M0_CLOBBER)
        else:
            # s_add_u32 writes SCC: without the clobber the compiler keeps a loop compare's
            # SCC live across the asm (the tile loop then exits after one tile)
            asm = (
                f"s_add_u32 m0, {stride}, m0\nbuffer_load_dwordx4 $0, $1, $2 offen lds"
            )
            _asm_void([voff, self.rsrc, soff], asm, "v,s,s", "~{scc}")

    def load(self, lds_dst, k_offset):
        for step in range_constexpr(self.n_load_steps):
            self._emit(lds_dst, k_offset, step)

    def load_one(self, lds_dst, k_offset, step):
        self._emit(lds_dst, k_offset, step)


class S2RLoader:
    """LDS -> the 32-byte MFMA operand of lane (row lane % 16, klane lane // 16): the two
    16-byte K segments at 16 * klane and 64 + 16 * klane of the 128-byte row."""

    def __init__(self, wave_idx, n_tiles):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles

    def _vec_load_16(self, lds_src, dyn_offset, const_offset):
        total_off = lds_src.byte_off + const_offset
        window_base = (total_off // 0x10000) * 0x10000
        imm = total_off - window_base
        assert 0 <= imm <= 0xFFFF
        vaddr = fx.Int32(fx.ptrtoint(lds_src.base_ptr)) + fx.Int32(
            window_base + dyn_offset
        )
        lds_ptr = _llvm.inttoptr(_lds_ptr_t(), fx.as_ir_value(vaddr))
        if imm != 0:
            lds_ptr = _gep(lds_ptr, static_byte_offset=imm)
        vec4_i32 = _ir.VectorType.get([4], fx.Int32.ir_type)
        return Vec(_llvm.LoadOp(vec4_i32, lds_ptr, alignment=16).result)

    def _dyn_offset(self, seg, preshuffled):
        row = self.wave_idx * (self.n_tiles * 16) + self.lane_id % 16
        col = (self.lane_id // 16) * 16 + seg * 64
        if const_expr(preshuffled):
            return (row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128
        row_swz, col_swz = _swizzle_128(row, col)
        return row_swz * 128 + col_swz

    def load_seg(self, lds_src, i, seg, preshuffled=False):
        """One 16-byte segment (tile i, K segment seg) as i32x4."""
        return self._vec_load_16(
            lds_src, self._dyn_offset(seg, preshuffled), i * 2048
        ).bitcast(fx.Int32)

    def load(self, lds_src, preshuffled=False):
        return [_pack8(self.load_seg(lds_src, i, 0, preshuffled), self.load_seg(lds_src, i, 1, preshuffled))
                for i in range_constexpr(self.n_tiles)]  # fmt: skip


def _pack8(lo, hi):
    return lo.shuffle(hi, list(range(8)))


def _g2s_thunks(g2s, dst, gl_off, n_steps):
    return [lambda s=s: g2s.load_one(dst, gl_off, s) for s in range(n_steps)]


def _riffle(glb, lds):
    """Interleave the global and LDS thunk lists proportionally."""
    if not glb or not lds:
        return list(glb) + list(lds)
    out = []
    step = len(lds) / len(glb)
    li = 0
    for gi, t in enumerate(glb):
        out.append(t)
        upto = round((gi + 1) * step)
        out += lds[li:upto]
        li = upto
    return out + lds[li:]


def _s2r_thunks(s2r, src, holder, n, pre):
    """Thunks loading tile i's two segments into holder[i] (packed after the second)."""
    ts = []
    for i in range(n):
        for seg in range(2):

            def f(i=i, seg=seg):
                v = s2r.load_seg(src, i, seg, preshuffled=pre)
                if seg == 0:
                    holder[i] = v
                else:
                    holder[i] = _pack8(holder[i], v)

            ts.append(f)
    return ts


def _min(a, b):
    return fx.arith.select(a < b, a, b)


def _divmod_nonneg(a, b):
    if const_expr(isinstance(b, int) and b > 0 and (b & (b - 1)) == 0):
        sh = b.bit_length() - 1
        return (a >> sh, a & (b - 1)) if const_expr(sh > 0) else (a, 0)
    return (a // b, a % b)


def _xcd_swizzle(num_pid_m, num_pid_n, wgid):
    """Tile index -> (tile_i, tile_j), aiter's XCD-aware grouping for any grid: workgroup
    ``w`` runs on XCD ``w % 8``, so XCD ``x`` gets the indices ``8 q + x`` and walks its
    own eighth of the tiles in groups of 4 rows, column-major -- its 32 CUs hold about 8
    column tiles of W and 4 row tiles of A at a time. (The row-major order below 1024
    tiles spread an XCD over every column unless the column count was a multiple of 8:
    N 9856 = 39 columns ran 10% slower than N 10240 = 40 at M 4096.) An index beyond the
    grid lands on row ``num_pid_m``, beyond M, where the buffer bounds drop its stores.
    """
    NUM_XCDS = 8
    WGM = 4
    num_wg = num_pid_m * num_pid_n
    in_grid = (wgid >= 0) & (wgid < num_wg)
    wgid = fx.arith.select(in_grid, wgid, 0)
    intra_xcd, xcd = _divmod_nonneg(wgid, NUM_XCDS)
    base, extra = _divmod_nonneg(num_wg, NUM_XCDS)
    wgid_remap = xcd * base + _min(xcd, extra) + intra_xcd
    group_id, intra_group = _divmod_nonneg(wgid_remap, WGM * num_pid_n)
    first_pid_m = group_id * WGM
    group_size_m = _min(num_pid_m - first_pid_m, WGM)
    pid_n, intra_group_m = _divmod_nonneg(intra_group, group_size_m)
    pid_m = fx.arith.select(in_grid, first_pid_m + intra_group_m, num_pid_m)
    return pid_m, pid_n


class MfmaRowStationary:
    """The scaled fp8 16x16x128 MFMA with the accumulator pinned in AGPRs (``=a`` / tied
    ``0``). Operands are fed (W, A) so lane (klane, r) accumulates out[row r, 4 klane ..]
    (C^T = W A^T); ``sa`` / ``sb`` are the packed e8m0 dwords of the 32-row / 32-column
    groups, op_sel = (tile parity) picks the row half, op_sel_hi = (K-step parity) the
    K half. A quadrant issues row by row (``row_stationary_order``, the order of the MoE
    gemm1/gemm2 PP kernels' ``MfmaRowStationary``)."""

    def __init__(self, n_a, n_b):
        self.res_ty = Vec.make_type(4, fx.Float32)
        self.n_a = n_a
        self.n_b = n_b

    def idx(self, i, j):
        return i * self.n_b + j

    def _order(self):
        return row_stationary_order(self.n_a, self.n_b)

    def call(self, a, b, c, sa, sb, khalf, interleave=None, zero_acc=False):
        thunks = list(interleave) if interleave else []
        nth = [0]
        mth = [0]
        order = self._order()
        n_mfma = len(order)
        slots = (
            {(t * n_mfma) // len(thunks) for t in range(len(thunks))}
            if thunks
            else set()
        )
        for i, j in order:
            acc = None if zero_acc else c[self.idx(i, j)]
            c[self.idx(i, j)] = self._mfma(
                a[i],
                b[j],
                acc,
                sa[i // _PACK],
                sb[j // _PACK],
                khalf,
                i % _PACK,
                j % _PACK,
            )
            if nth[0] < len(thunks) and mth[0] in slots:
                thunks[nth[0]]()
                nth[0] += 1
            mth[0] += 1
        while nth[0] < len(thunks):
            thunks[nth[0]]()
            nth[0] += 1
        return c

    def _mfma(self, a_op, b_op, acc, sa_v, sb_v, khalf, ia, jb):
        # (W, A) order: swap the roles
        a_op, b_op = b_op, a_op
        sa_v, sb_v = sb_v, sa_v
        ia, jb = jb, ia
        src2 = "$0" if acc is not None else "0"
        asm = (f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, {src2}, $3, $4 "
               f"op_sel:[{ia},{jb},0] op_sel_hi:[{khalf},{khalf},0] cbsz:0 blgp:0")  # fmt: skip
        ops = [
            fx.as_ir_value(a_op),
            fx.as_ir_value(b_op),
            fx.as_ir_value(sa_v),
            fx.as_ir_value(sb_v),
        ]
        cons = "=a,v,v,v,v"
        if acc is not None:
            ops.append(fx.as_ir_value(acc))
            cons += ",0"
        return _llvm.inline_asm(self.res_ty, ops, asm, cons, has_side_effects=True)


_SCALE_QUARTER_BYTES = 1024  # one wave's gather: 4 groups x 256 B of one 256-K block
_SCALE_REGION_BYTES = 2 * _SCALE_QUARTER_BYTES  # all of A (or all of W) for one block
_SCALE_SLOT_BYTES = _N_WAVES * _SCALE_QUARTER_BYTES  # 4 KB
_SCALE_SLOTS = 2  # the block in use (VGPR carry) and the next one
_SCALE_A_REGION = 0
_SCALE_B_REGION = _SCALE_REGION_BYTES


class ScaleGatherLDS:
    """Per 256-K block one ``buffer_load_dwordx4 ... lds`` per wave: waves 0 / 1 fetch
    the A scales of their row quarters (groups {0, 1, 4, 5} and {2, 3, 6, 7} of the tile's
    eight 32-row groups -- each wave's own rows), waves 2 / 3 the W scales alike."""

    def __init__(
        self,
        a_scale,
        b_scale,
        K,
        lane_id,
        wave_id,
        lds_base_ptr,
        a_rows,
        b_rows,
        n_a,
        n_b,
    ):
        self.row_i32 = (K // 256) * 64  # dwords per 32-row group
        self.wave_id = wave_id
        self.n_a = (
            n_a  # 16-row tiles per wave quadrant: the wave's quarter is 16 n_a rows,
        )
        self.n_b = n_b  # a 128-row half of the parent layout holds n_a groups
        row_bytes = K // 32
        # the shuffled scale layout interleaves a 32-row group, so the bound is whole groups
        # (the quantizer pads the rows to 32; a bound of M rows zeroes the last group's scales)
        a_rows = (a_rows + 31) // 32 * 32
        # aiter's create_buffer_resource returns the raw !llvm.ptr<8> resource
        self.a_rsrc = fx.as_ir_value(
            _buffer_ops.create_buffer_resource(
                a_scale, max_size=False, num_records_bytes=a_rows * row_bytes
            )
        )
        self.b_rsrc = fx.as_ir_value(
            _buffer_ops.create_buffer_resource(
                b_scale, max_size=False, num_records_bytes=b_rows * row_bytes
            )
        )
        self._blk = lane_id // 16
        self._in16 = lane_id % 16
        self._lds_base = fx.Int32(fx.ptrtoint(lds_base_ptr))

    def set_wave_base(self):
        wid = fx.Int32(_rocdl.readfirstlane(_T.i32, fx.as_ir_value(self.wave_id)))
        self._wave_base_s = fx.as_ir_value(
            self._lds_base + wid * fx.Int32(_SCALE_QUARTER_BYTES)
        )
        self._is_a = wid < fx.Int32(2)
        self._q = wid % fx.Int32(2)
        self._rsrc = fx.arith.select(self._is_a, self.a_rsrc, self.b_rsrc)
        self._soff0 = _uniform_i32(fx.Int32(0))

    def tile_group(self, a_base_tile, b_base_tile):
        """This wave's first 32-row group of a tile (wave-uniform)."""
        base_tile = fx.arith.select(
            self._is_a,
            a_base_tile + self._q * fx.Int32(16 * self.n_a),
            b_base_tile + self._q * fx.Int32(16 * self.n_b),
        )
        return _uniform_i32(base_tile // fx.Int32(32))

    def set_tile_voff(self, G):
        """This tile's per-lane gather offset with the K-block folded out (it rides
        the uniform soffset instead): computed once per tile, so the K loop carries
        no per-gather voffset math. Lane blocks 0 / 1 = the wave's groups of the first
        tile half, 2 / 3 of the second (a half holds n_a / n_b groups); blocks beyond the
        wave's groups fetch neighbours."""
        half = fx.arith.select(self._is_a, fx.Int32(self.n_a), fx.Int32(self.n_b))
        grp = fx.Int32(G) + (self._blk // 2) * half + (self._blk % 2)
        self._voff_g = fx.as_ir_value(
            (grp * fx.Int32(self.row_i32) + self._in16 * fx.Int32(4)) * fx.Int32(4)
        )

    def gather_soff(self, m0_s, soff):
        """The gather of one scale block: the K-block byte offset in the uniform soffset,
        the slot's LDS address precomputed (m0_s)."""
        asm = "s_mov_b32 m0, $0\nbuffer_load_dwordx4 $1, $2, $3 offen lds"
        soff_v = fx.Int32(soff) if isinstance(soff, int) else soff
        _asm_void(
            [fx.as_ir_value(m0_s), self._voff_g, self._rsrc, _uniform_i32(soff_v)],
            asm,
            "s,v,s,s",
            _M0_CLOBBER,
        )


class ScaleLoaderLDS:
    """The wave's scale dwords of a block: lane (klane, r) reads the dword of group
    (half, gi) at its (klane, mlane) position -- the layout the MFMA wants."""

    def __init__(self, n_tiles, lane_id, quarter, lds_base_ptr, region_off):
        assert n_tiles % _PACK == 0
        self.n_groups = n_tiles // _PACK
        self.lane_id = lane_id
        self._region_base = (
            fx.Int32(fx.ptrtoint(lds_base_ptr))
            + fx.Int32(region_off)
            + quarter * fx.Int32(_SCALE_QUARTER_BYTES)
        )

    def read_half(self, slot, half):
        L = self.lane_id
        base = (
            self._region_base
            + fx.Int32(slot) * fx.Int32(_SCALE_SLOT_BYTES)
            + fx.Int32((L // 4) * 16 + (L % 4) * 4)
        )
        out = []
        for gi in range_constexpr(self.n_groups):
            blk = half * 2 + gi
            lds_ptr = _llvm.inttoptr(
                _lds_ptr_t(), fx.as_ir_value(base + fx.Int32(blk * 256))
            )
            out.append(
                fx.Int32(_llvm.LoadOp(fx.Int32.ir_type, lds_ptr, alignment=4).result)
            )
        return out

    def read(self, slot):
        return self.read_half(slot, 0), self.read_half(slot, 1)


def _acc_read(vec4, i):
    """``v_accvgpr_read_b32`` as inline asm: the compiler does not know the inline-asm
    MFMA's latency and would hoist a plain read up to the MFMA that writes the
    accumulator (a stale read: the first two components of a tile came out wrong); a
    side-effecting asm keeps its place in the stream, after enough later MFMAs."""
    el = fx.as_ir_value(Vec(vec4)[i])
    return fx.Float32(
        _llvm.inline_asm(
            fx.Float32.ir_type,
            [el],
            "v_accvgpr_read_b32 $0, $1",
            "=v,a",
            has_side_effects=True,
        )
    )


class StoreCLds:
    """bf16 output through a per-wave LDS transpose (2 buffers): a block of 16 rows x
    (16 n_b) columns of a quadrant is written in the MFMA layout (8 bytes per lane per
    tile: 4 columns of one row) and read back as slabs of (32 / n_b) rows x (32 n_b) bytes,
    so every ``buffer_store_dwordx4`` writes whole rows (8 columns per lane, 2 n_b lanes
    per row) instead of 64 scattered 16-byte pieces (consecutive lanes are consecutive
    rows in the MFMA layout, and the store path handles those pieces one by one: ~3x the
    issue cost); no permlane swaps and their wait states. The 16-byte pieces of a row are
    XOR-swizzled by (row % pieces) against LDS bank conflicts. (Writing the f32
    accumulators straight from the AGPRs with ``ds_write_b128`` and converting on the read
    side was measured too: no faster, twice the LDS bytes.)"""

    def __init__(
        self,
        C,
        c_rows,
        c_cols,
        c_idx_fn,
        lds_base_ptr,
        wave_id,
        n_a,
        n_b,
        nt=False,
    ):
        self.c_cols = c_cols
        self.c_rows = c_rows
        self.c_idx_fn = c_idx_fn
        self.n_a = n_a
        self.n_b = n_b
        self.pieces = 2 * n_b  # 16-byte pieces per block row (32 n_b bytes)
        self.rows_per_read = 64 // self.pieces  # rows one 64-lane read / store covers
        self.slabs = 16 // self.rows_per_read  # reads per block
        self.buf_bytes = 16 * 32 * n_b
        self.cpol = (
            2 if nt else 0
        )  # gfx950 buffer_store bits: 0x2 = nt (streaming, write-once output)
        self.rsrc = _buffer_ops.create_buffer_resource(
            C, max_size=False, num_records_bytes=c_rows * c_cols * 2
        )
        lane = fx.thread_idx.x % 64
        g = lane // 16
        r = lane % 16
        row_b = 32 * n_b
        base = fx.Int32(fx.ptrtoint(lds_base_ptr)) + wave_id * fx.Int32(
            2 * self.buf_bytes
        )
        # write side: lane (g, r) holds columns 4 g .. 4 g + 3 of row r of each 16x16 tile
        self.w_off = [
            base
            + r * fx.Int32(row_b)
            + (((fx.Int32(2 * tj) + g // 2) ^ (r % self.pieces)) * fx.Int32(16))
            + (g % 2) * fx.Int32(8)
            for tj in range_constexpr(n_b)
        ]
        # read side: lane (r2 = lane // pieces, q = lane % pieces) takes piece q of row r2 of each slab
        q = lane % self.pieces
        r2 = lane // self.pieces
        self.r_off = [
            base
            + (r2 + h * self.rows_per_read) * fx.Int32(row_b)
            + ((q ^ (r2 % self.pieces)) * fx.Int32(16))
            for h in range_constexpr(self.slabs)
        ]
        self.st_off = (
            r2 * c_cols + q * 8
        ) * 2  # the per-lane part of the store byte offset

    @property
    def n_thunks(self):
        return self.n_a * (1 + self.slabs)

    def _write(self, c_frag, ti, tj, buf):
        v = c_frag[self.c_idx_fn(ti, tj)]
        rd = _acc_read
        d0 = _cvt_pk_bf16(rd(v, 0), rd(v, 1))
        d1 = _cvt_pk_bf16(rd(v, 2), rd(v, 3))
        vec = Vec.from_elements([fx.Int32(d0), fx.Int32(d1)], fx.Int32)
        ptr = _llvm.inttoptr(
            _lds_ptr_t(),
            fx.as_ir_value(self.w_off[tj] + fx.Int32(buf * self.buf_bytes)),
        )
        _llvm.StoreOp(fx.as_ir_value(vec), ptr, alignment=8)

    def _read_store(self, base_row, base_col, ti, h, buf):
        ptr = _llvm.inttoptr(
            _lds_ptr_t(), fx.as_ir_value(self.r_off[h] + fx.Int32(buf * self.buf_bytes))
        )
        vec4 = _ir.VectorType.get([4], fx.Int32.ir_type)
        data = Vec(_llvm.LoadOp(vec4, ptr, alignment=16).result)
        off = (
            (base_row + ti * 16 + h * self.rows_per_read) * self.c_cols + base_col
        ) * 2 + self.st_off
        _buffer_ops.buffer_store(
            data, self.rsrc, off, offset_is_bytes=True, cache_modifier=self.cpol
        )

    def thunks(self, c_frag, base_row, base_col):
        """n_a (1 + slabs) thunks per quadrant: the blocks through the two buffers -- W0 W1
        R0 W2 R1 ... (a wave's LDS ops run in order, so a buffer's re-write after its reads
        needs no wait); quadrants sharing the buffers must be concatenated, not
        interleaved."""
        order = [("w", 0)]
        for b in range(1, self.n_a):
            order += [("w", b), ("r", b - 1)]
        order.append(("r", self.n_a - 1))
        ts = []
        for kind, ti in order:
            if kind == "w":
                ts.append(
                    lambda ti=ti: [
                        self._write(c_frag, ti, tj, ti % 2) for tj in range(self.n_b)
                    ]
                )
            else:
                for h in range(self.slabs):
                    ts.append(
                        lambda ti=ti, h=h: self._read_store(
                            base_row, base_col, ti, h, ti % 2
                        )
                    )
        return ts

    def store(self, c_frag, base_row, base_col):
        for t in self.thunks(c_frag, base_row, base_col):
            t()


def compile_mxfp8_gemm_4w(
    *,
    K: int,
    use_xcd_remap: bool = True,
    store_nt: bool = False,
    tiles: tuple = (4, 4),
    grid_max: int = 256,
):
    """``store_nt`` = non-temporal output stores (pays when the output is larger than the
    256 MB infinity cache: o_proj at M 32768 -2.5%, nothing at qkv); ``tiles`` = (n_a, n_b)
    16-row / 16-column MFMA tiles per wave quadrant: (4, 4) is the 256x256 tile, (2, 4) a
    128x256 and (2, 2) a 128x128 sub-tile of it -- the launch takes a range of work items
    ``(tile_off, n_work)`` over the 256x256 parent tiles' index space (each parent = 1 / 2 / 4
    items), so a remainder round of parent tiles can run as one full round of sub-tiles.

    Each wave quadrant issues its MFMAs in the row-stationary order (``MfmaRowStationary``);
    the g2s / scale-gather addresses ride uniform accumulators instead of being recomputed
    every K-step.

    Persistent: the grid is min(tiles, 256) workgroups and workgroup w walks the tiles
    w, w + grid, ...; the global loads of a tile's steps 0 / 1 and the gather of its scale
    block 0 go out in the last two steps of the previous tile (the pipeline never drains),
    the stores of a tile's c00 / c01 quadrants are interleaved with its last step's c10 /
    c11 MFMAs and those of c10 / c11 with the next tile's first c00 / c01 MFMAs (the old
    and the new accumulators share the 256 AGPRs half and half). Scale blocks are
    gathered one block ahead (at even steps) and read at the following odd step."""
    N_TILES_A, N_TILES_B = tiles
    assert tiles in ((4, 4), (2, 4), (2, 2)), tiles  # the layouts ``launch_plan`` runs
    BLOCK_M = 64 * N_TILES_A
    BLOCK_N = 64 * N_TILES_B
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    SUB_M = PARENT_M // BLOCK_M  # sub-tiles per parent tile
    SUB_N = PARENT_N // BLOCK_N
    SUB_SPLIT = SUB_M * SUB_N
    assert K % 256 == 0 and K >= 512, K  # the K loop needs >= 4 steps
    K_ITERS = K // BLOCK_K  # even
    UNROLL = 4 if (K_ITERS - 4) % 4 == 0 else 2
    assert (K_ITERS - 4) % UNROLL == 0, K_ITERS
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert (
        LDS_BLOCK_M * BLOCK_K <= _LDS_BUF_BYTES
        and LDS_BLOCK_N * BLOCK_K <= _LDS_BUF_BYTES
    )
    # grid_max: workgroups of the launch (256 = one per CU, persistent); a larger value gives
    # every work item its own workgroup (a plain grid), which no co-resident kernel holds back
    GRID_MAX = grid_max
    # v5: the row-stationary MFMA order with the scale-gather address accumulators
    name = (f"flydsl_mxfp8_4wave_v5_k{K}_xcd{int(bool(use_xcd_remap))}" + ("_nt" if store_nt else "") + ("" if grid_max == 256 else f"_g{grid_max}")
            + (f"_t{N_TILES_A}x{N_TILES_B}" if tiles != (4, 4) else ""))  # fmt: skip
    SCALE_SLOTS = _SCALE_SLOTS
    OUT_LDS_BYTES = _N_WAVES * 2 * 16 * 32 * N_TILES_B

    @fx.struct
    class SharedStorage:
        all_lds: fx.Array[fx.Int8, 8 * _LDS_BUF_BYTES, 16]
        scale_lds: fx.Array[fx.Int8, SCALE_SLOTS * _SCALE_SLOT_BYTES, 16]
        out_lds: fx.Array[fx.Int8, OUT_LDS_BYTES, 16]

    @flyc.kernel(name=name)
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        tile_off: fx.Int32,
        n_work: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        _base_ptr = lds.all_lds.ptr

        def _buf(idx):
            return _Buf(_base_ptr, idx * _LDS_BUF_BYTES)

        a_cur0, a_cur1, a_next0, a_next1 = _buf(0), _buf(1), _buf(2), _buf(3)
        b_cur0, b_cur1, b_next0, b_next1 = _buf(4), _buf(5), _buf(6), _buf(7)

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        n_pid_m = _ceildiv(c_m, PARENT_M)  # the parent tiles' grid
        n_blocks = _ceildiv(c_n, PARENT_N)
        grid = _min(fx.Int32(n_work), fx.Int32(GRID_MAX))
        wg = fx.Int32(fx.block_idx.x)
        my_tiles = (fx.Int32(n_work) - wg + grid - fx.Int32(1)) // grid
        wave_i = wave_id // 2
        wave_j = wave_id % 2
        A_K_STEP = BLOCK_K
        B_K_STEP = 2 * 1024  # preshuffled (16, 16): a 16-row block spans 2 KB per 128 K

        def tile_coords(t):
            """work item -> (tile_i, tile_j) in units of this kernel's tile. The parent tiles
            are indexed in the XCD order of ``_xcd_swizzle``; a sub-tile launch keeps a
            workgroup on its XCD's parents: item ``t = 8 q + x`` is sub-tile ``q % SUB_SPLIT``
            of parent ``tile_off + 8 (q / SUB_SPLIT) + x`` (``tile_off`` a multiple of 8). A
            negative / out-of-range index lands beyond M so its stores are dropped by the
            buffer bounds."""
            if const_expr(use_xcd_remap and SUB_SPLIT > 1):
                q, x = _divmod_nonneg(t, 8)
                p_intra, sub = _divmod_nonneg(q, SUB_SPLIT)
                parent = fx.Int32(tile_off) + p_intra * 8 + x
            else:
                parent, sub = _divmod_nonneg(t, SUB_SPLIT)
                parent = fx.Int32(tile_off) + parent
            if const_expr(use_xcd_remap):
                pi, pj = _xcd_swizzle(n_pid_m, n_blocks, parent)
            else:
                pi, pj = parent // n_blocks, parent % n_blocks
            si, sj = _divmod_nonneg(sub, SUB_N)
            return pi * SUB_M + si, pj * SUB_N + sj

        def tile_bases(ti, tj):
            """the g2s bases of a tile (row-major A, preshuffled W) and its scale groups"""
            return (
                (ti * BLOCK_M) * K,
                (ti * BLOCK_M + LDS_BLOCK_M) * K,
                (tj * BLOCK_N) * K,
                (tj * BLOCK_N + LDS_BLOCK_N) * K,
            )

        mfma = MfmaRowStationary(N_TILES_A, N_TILES_B)
        _scale_base_ptr = lds.scale_lds.ptr
        scale_gather = ScaleGatherLDS(
            A_scale,
            B_scale,
            K,
            lane_id,
            wave_id,
            _scale_base_ptr,
            c_m,
            c_n,
            N_TILES_A,
            N_TILES_B,
        )
        scale_gather.set_wave_base()
        # the two scale-slot LDS addresses (slot 0 / 1) precomputed once; the
        # gather's K-block offset rides the uniform soffset inside the K loop
        _gm0_s = (
            scale_gather._wave_base_s,
            fx.as_ir_value(
                fx.Int32(scale_gather._wave_base_s) + fx.Int32(_SCALE_SLOT_BYTES)
            ),
        )
        a_scale_ld = ScaleLoaderLDS(
            N_TILES_A, lane_id, wave_i, _scale_base_ptr, _SCALE_A_REGION
        )
        b_scale_ld = ScaleLoaderLDS(
            N_TILES_B, lane_id, wave_j, _scale_base_ptr, _SCALE_B_REGION
        )

        def _slot(b):
            return fx.Int32(b) % fx.Int32(SCALE_SLOTS)

        gl_off_a = _global_swizzle(lane_id, wave_id, K, N_TILES_A, False)
        gl_off_b = _global_swizzle(lane_id, wave_id, K, N_TILES_B, True)
        a_rsrc = _buffer_ops.create_buffer_resource(
            A, max_size=False, num_records_bytes=c_m * K
        )
        b_rsrc = _buffer_ops.create_buffer_resource(
            B_T, max_size=False, num_records_bytes=c_n * K
        )
        a_g2s = G2SLoaderAsm(a_rsrc, gl_off_a, N_TILES_A, wave_id)
        b_g2s = G2SLoaderAsm(b_rsrc, gl_off_b, N_TILES_B, wave_id)
        a_g2s.set_wave_base(_base_ptr)
        b_g2s.set_wave_base(_base_ptr)
        a_s2r = S2RLoader(wave_i, N_TILES_A)
        b_s2r = S2RLoader(wave_j, N_TILES_B)
        store_c = StoreCLds(
            C,
            c_m,
            c_n,
            mfma.idx,
            lds.out_lds.ptr,
            wave_id,
            N_TILES_A,
            N_TILES_B,
            nt=store_nt,
        )

        # a step issues [ac0 (n_a) | bc0 (n_b) | bc1 (n_b) | ac1 (n_a)] loads per wave: at a
        # step's top the previous step's may all be in flight; mid-step the previous step's
        # ac1 and this step's first two batches
        _MAIN_VMCNT = 2 * N_TILES_A + 2 * N_TILES_B
        _SEG2_VMCNT = 2 * N_TILES_A + N_TILES_B

        def _read_scale_thunks(block, holder):
            # the caller passes the LDS slot itself (the block's K offset rides the
            # uniform soffset, so no block -> slot math is left)
            s = block

            def _r(dst, ld, half, _s=s):
                holder[dst] = ld.read_half(_s, half)

            return [
                lambda: _r(0, a_scale_ld, 0),
                lambda: _r(1, a_scale_ld, 1),
                lambda: _r(2, b_scale_ld, 0),
                lambda: _r(3, b_scale_ld, 1),
            ]

        def _g2s_bases(bases, step):
            a0, a1, b0, b1 = bases
            st = fx.Int32(step)
            return (
                fx.Int32(a0) + st * fx.Int32(A_K_STEP),
                fx.Int32(a1) + st * fx.Int32(A_K_STEP),
                fx.Int32(b0) + st * fx.Int32(B_K_STEP),
                fx.Int32(b1) + st * fx.Int32(B_K_STEP),
            )

        def _one_step(
            parity,
            a0f,
            b0f,
            b1f_in,
            sc,
            accs,
            bufs,
            load,
            gather,
            read_block,
            zero_acc=False,
            extra=None,
        ):
            """One K-step: this step's 64 MFMAs with, interleaved, the LDS reads for the next
            step, the global loads ``load`` of the step after (the resolved (a0, a1, b0,
            b1) soffsets), at even steps the gather ``gather`` of the next scale block (the
            precomputed (m0, soff)), at odd steps the read of scale block ``read_block``
            (the LDS slot itself) into the VGPR carry;
            ``extra(g, c00f, c01f)`` (called before group g = 0..3 with the accumulators
            finished so far, None before) returns more thunks for that group: the stores
            of this tile's c00 / c01 in the last step, of the previous tile's c10 / c11
            in the first."""
            ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
            saR0, saR1, sbC0, sbC1 = sc
            c00f, c01f, c10f, c11f = accs
            _a1 = [None] * N_TILES_A
            _a0n = [None] * N_TILES_A
            _b0n = [None] * N_TILES_B
            _b1n = [None] * N_TILES_B
            a0_off, a1_off, b0_off, b1_off = load
            _scn = [None, None, None, None]
            _rd_scn = _read_scale_thunks(read_block, _scn) if parity == 1 else []
            _sc_gather = (
                []
                if parity == 1
                else [lambda: scale_gather.gather_soff(gather[0], gather[1])]
            )
            ex = extra if extra is not None else (lambda g, c00, c01: ())

            wait_barrier(_MAIN_VMCNT)
            il = _riffle(
                _g2s_thunks(a_g2s, ac0, a0_off, N_TILES_A),
                _s2r_thunks(a_s2r, ac1, _a1, N_TILES_A, False),
            ) + list(ex(0, None, None))
            c00f = mfma.call(
                a0f, b0f, c00f, saR0, sbC0, parity, interleave=il, zero_acc=zero_acc
            )
            il = _riffle(
                _g2s_thunks(b_g2s, bc0, b0_off, N_TILES_B), list(ex(1, c00f, None))
            )
            c01f = mfma.call(
                a0f, b1f_in, c01f, saR0, sbC1, parity, interleave=il, zero_acc=zero_acc
            )
            a1f = _a1

            wait_barrier(_SEG2_VMCNT)
            il = (_riffle(_g2s_thunks(b_g2s, bc1, b1_off, N_TILES_B), _s2r_thunks(a_s2r, an0, _a0n, N_TILES_A, False))
                  + _sc_gather + _rd_scn[:2] + list(ex(2, c00f, c01f)))  # fmt: skip
            c10f = mfma.call(
                a1f, b0f, c10f, saR1, sbC0, parity, interleave=il, zero_acc=zero_acc
            )
            a0nf = _a0n
            il = (
                _riffle(
                    _g2s_thunks(a_g2s, ac1, a1_off, N_TILES_A),
                    _s2r_thunks(b_s2r, bn0, _b0n, N_TILES_B, True)
                    + _s2r_thunks(b_s2r, bn1, _b1n, N_TILES_B, True),
                )
                + _rd_scn[2:]
                + list(ex(3, c00f, c01f))
            )
            c11f = mfma.call(
                a1f, b1f_in, c11f, saR1, sbC1, parity, interleave=il, zero_acc=zero_acc
            )
            b0nf, b1nf = _b0n, _b1n

            sc_next = (_scn[0], _scn[1], _scn[2], _scn[3]) if parity == 1 else sc
            new_bufs = (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)
            return a0nf, b0nf, b1nf, sc_next, (c00f, c01f, c10f, c11f), new_bufs

        bufs0 = (a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1)

        n_a = N_TILES_A
        n_b = N_TILES_B
        n_ga = N_TILES_A // _PACK
        n_gb = N_TILES_B // _PACK
        n_sc = 2 * n_ga + 2 * n_gb
        _R = fx.as_ir_value

        def _flat_sc(sc):
            saR0, saR1, sbC0, sbC1 = sc
            return (
                [_R(v) for v in saR0]
                + [_R(v) for v in saR1]
                + [_R(v) for v in sbC0]
                + [_R(v) for v in sbC1]
            )

        def _unflat_sc(flat):
            o = 0
            saR0 = list(flat[o : o + n_ga])
            o += n_ga
            saR1 = list(flat[o : o + n_ga])
            o += n_ga
            sbC0 = list(flat[o : o + n_gb])
            o += n_gb
            sbC1 = list(flat[o : o + n_gb])
            return (saR0, saR1, sbC0, sbC1)

        # ---- prologue: the first tile's steps 0 / 1 and its scale blocks 0 / 1 ----
        t0 = wg
        ti0, tj0 = tile_coords(t0)
        bases0 = tile_bases(ti0, tj0)
        G0 = scale_gather.tile_group(ti0 * BLOCK_M, tj0 * BLOCK_N)
        scale_gather.set_tile_voff(G0)
        scale_gather.gather_soff(_gm0_s[0], 0)
        scale_gather.gather_soff(_gm0_s[1], 256)
        s0 = _g2s_bases(bases0, 0)
        s1 = _g2s_bases(bases0, 1)
        a_g2s.load(a_cur0, s0[0])
        b_g2s.load(b_cur0, s0[2])
        b_g2s.load(b_cur1, s0[3])
        a_g2s.load(a_cur1, s0[1])
        a_g2s.load(a_next0, s1[0])
        b_g2s.load(b_next0, s1[2])
        b_g2s.load(b_next1, s1[3])
        a_g2s.load(a_next1, s1[1])
        # 2 gathers + 32 loads in flight: a_cur0 landed at vmcnt(28), b_cur0 / b_cur1 at 20
        wait_barrier((3 * N_TILES_A) + (4 * N_TILES_B))
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier((3 * N_TILES_A) + (2 * N_TILES_B))
        b0_frag = b_s2r.load(b_cur0, preshuffled=True)
        b1_frag = b_s2r.load(b_cur1, preshuffled=True)
        sc0_saR0, sc0_saR1 = a_scale_ld.read(_slot(0))
        sc0_sbC0, sc0_sbC1 = b_scale_ld.read(_slot(0))
        sc0 = (sc0_saR0, sc0_saR1, sc0_sbC0, sc0_sbC1)
        zero4 = Vec.filled(4, 0.0, fx.Float32)
        c10_prev = [zero4] * N_ACCUMS
        c11_prev = [zero4] * N_ACCUMS

        # ---- the tile loop: state = fragments of the tile's step 0, its scale block 0, the
        # previous tile's c10 / c11 (stored during this tile's first step), the tile index
        tile_state = (
            [_R(x) for x in a0_frag] + [_R(x) for x in b0_frag] + [_R(x) for x in b1_frag] + _flat_sc(sc0)
            + [_R(x) for x in c10_prev] + [_R(x) for x in c11_prev] + [_R(fx.Int32(t0)), _R(fx.Int32(-1))]
        )  # fmt: skip
        for jt, tstate in range(0, my_tiles, 1, init=tile_state):
            off = 0
            a0f = list(tstate[off : off + n_a])
            off += n_a
            b0f = list(tstate[off : off + n_b])
            off += n_b
            b1f = list(tstate[off : off + n_b])
            off += n_b
            sc = _unflat_sc(tstate[off : off + n_sc])
            off += n_sc
            c10p = list(tstate[off : off + N_ACCUMS])
            off += N_ACCUMS
            c11p = list(tstate[off : off + N_ACCUMS])
            off += N_ACCUMS
            t_cur = fx.Int32(tstate[off])
            t_prev = fx.Int32(tstate[off + 1])
            t_next = t_cur + grid
            ti, tj = tile_coords(t_cur)
            bases = tile_bases(ti, tj)
            G = scale_gather.tile_group(ti * BLOCK_M, tj * BLOCK_N)
            scale_gather.set_tile_voff(G)
            # the next tile (the last tile loads a tile beyond the grid: rows beyond M read
            # zero through the buffer bounds and nothing of it is stored)
            tin, tjn = tile_coords(t_next)
            bases_n = tile_bases(tin, tjn)
            Gn = scale_gather.tile_group(tin * BLOCK_M, tjn * BLOCK_N)
            # the previous tile's c10 / c11 store bases (row beyond M when there is none)
            tip, tjp = tile_coords(t_prev)
            has_prev = t_prev >= fx.Int32(0)
            prev_row1 = fx.arith.select(
                has_prev,
                tip * BLOCK_M + wave_i * (N_TILES_A * 16) + LDS_BLOCK_M,
                fx.Int32(c_m),
            )
            prev_col0 = fx.arith.select(
                has_prev, tjp * BLOCK_N + wave_j * (N_TILES_B * 16), fx.Int32(0)
            )
            prev_col1 = fx.arith.select(has_prev, prev_col0 + LDS_BLOCK_N, fx.Int32(0))
            base_row = ti * BLOCK_M + wave_i * (N_TILES_A * 16)
            base_col = tj * BLOCK_N + wave_j * (N_TILES_B * 16)

            accs = (
                [None] * N_ACCUMS,
                [None] * N_ACCUMS,
                [None] * N_ACCUMS,
                [None] * N_ACCUMS,
            )
            # step 0: zero accumulators, the previous tile's c10 / c11 stores spread over the
            # groups before the new c10 (born in group 2) / c11 (group 3), gather block 1
            prev_thunks = store_c.thunks(c10p, prev_row1, prev_col0) + store_c.thunks(
                c11p, prev_row1, prev_col1
            )
            n_prev3 = (
                len(prev_thunks) + 2
            ) // 3  # over the three groups before the new c11
            a0f, b0f, b1f, sc, accs, bufs = _one_step(
                0, a0f, b0f, b1f, sc, accs, bufs0, _g2s_bases(bases, 2), (_gm0_s[1], 256), None, zero_acc=True,
                extra=lambda g, c00, c01: prev_thunks[n_prev3 * g : n_prev3 * (g + 1)],
            )  # fmt: skip
            # step 1: read block 1
            a0f, b0f, b1f, sc, accs, bufs = _one_step(
                1, a0f, b0f, b1f, sc, accs, bufs, _g2s_bases(bases, 3), None, 1
            )
            # the four g2s streams advance by uniform strides and the gather
            # block by 256 B; carry them as uniform accumulators (incremented
            # per step) instead of recomputing base | (step << s) every K-step
            ka0 = fx.Int32(bases[0]) + fx.Int32(4 * A_K_STEP)
            ka1 = fx.Int32(bases[1]) + fx.Int32(4 * A_K_STEP)
            kb0 = fx.Int32(bases[2]) + fx.Int32(4 * B_K_STEP)
            kb1 = fx.Int32(bases[3]) + fx.Int32(4 * B_K_STEP)
            kg = fx.Int32(512)  # gather block soff at kk=2: block 2 << 8
            init_state = (
                [_R(x) for x in a0f] + [_R(x) for x in b0f] + [_R(x) for x in b1f] + _flat_sc(sc)
                + [_R(x) for x in accs[0]] + [_R(x) for x in accs[1]] + [_R(x) for x in accs[2]] + [_R(x) for x in accs[3]]
                + [_R(ka0), _R(ka1), _R(kb0), _R(kb1), _R(kg)]
            )  # fmt: skip
            for kk, state in range(2, K_ITERS - 2, UNROLL, init=init_state):
                off = 0
                a0f = list(state[off : off + n_a])
                off += n_a
                b0f = list(state[off : off + n_b])
                off += n_b
                b1f = list(state[off : off + n_b])
                off += n_b
                sc = _unflat_sc(state[off : off + n_sc])
                off += n_sc
                c00f = list(state[off : off + N_ACCUMS])
                off += N_ACCUMS
                c01f = list(state[off : off + N_ACCUMS])
                off += N_ACCUMS
                c10f = list(state[off : off + N_ACCUMS])
                off += N_ACCUMS
                c11f = list(state[off : off + N_ACCUMS])
                accs = (c00f, c01f, c10f, c11f)
                bufs = bufs0
                off += N_ACCUMS  # past c11f: the uniform address accumulators
                ka0 = fx.Int32(state[off])
                ka1 = fx.Int32(state[off + 1])
                kb0 = fx.Int32(state[off + 2])
                kb1 = fx.Int32(state[off + 3])
                kg = fx.Int32(state[off + 4])
                for u in range_constexpr(UNROLL):
                    if const_expr(UNROLL == 4):
                        # kk == 2 (mod 4): u=0 gathers an even block (slot 0), u=2 an
                        # odd one (slot 1) -- gather m0 slots folded; read slots stay
                        # runtime (see below).
                        lds_off = (ka0, ka1, kb0, kb1)
                        gath = None
                        if const_expr(u % 2 == 0):
                            gath = (
                                _gm0_s[u // 2],
                                kg if u == 0 else kg + fx.Int32(256),
                            )
                        rslot = None
                        if const_expr(u % 2 == 1):
                            # keep the read slot a runtime value: folding it to a
                            # constant made LLVM un-pair the scale ds_read2st64
                            # into 2 ds_read_b32 (+2 LDS/step at the ISA gate)
                            rslot = (
                                fx.Int32(kk) // fx.Int32(2) + fx.Int32(1 + u // 2)
                            ) % fx.Int32(2)
                    else:
                        kc = fx.Int32(kk) + fx.Int32(u)
                        block = kc // fx.Int32(2) + fx.Int32(1)
                        lds_off = _g2s_bases(bases, kc + fx.Int32(2))
                        gath = None
                        if const_expr(u % 2 == 0):
                            gath = (
                                fx.as_ir_value(
                                    fx.Int32(scale_gather._wave_base_s)
                                    + (block % fx.Int32(2))
                                    * fx.Int32(_SCALE_SLOT_BYTES)
                                ),
                                block * fx.Int32(256),
                            )
                        rslot = None
                        if const_expr(u % 2 == 1):
                            rslot = block % fx.Int32(2)
                    a0f, b0f, b1f, sc, accs, bufs = _one_step(
                        u % 2,
                        a0f,
                        b0f,
                        b1f,
                        sc,
                        accs,
                        bufs,
                        lds_off,
                        gath,
                        rslot,
                    )
                    ka0 = ka0 + fx.Int32(A_K_STEP)
                    ka1 = ka1 + fx.Int32(A_K_STEP)
                    kb0 = kb0 + fx.Int32(B_K_STEP)
                    kb1 = kb1 + fx.Int32(B_K_STEP)
                new_state = (
                    [_R(x) for x in a0f] + [_R(x) for x in b0f] + [_R(x) for x in b1f] + _flat_sc(sc)
                    + [_R(x) for x in accs[0]] + [_R(x) for x in accs[1]] + [_R(x) for x in accs[2]] + [_R(x) for x in accs[3]]
                    + [_R(ka0), _R(ka1), _R(kb0), _R(kb1), _R(kg + fx.Int32(UNROLL * 128))]
                )  # fmt: skip
                state = yield new_state

            off = 0
            a0f = list(state[off : off + n_a])
            off += n_a
            b0f = list(state[off : off + n_b])
            off += n_b
            b1f = list(state[off : off + n_b])
            off += n_b
            sc = _unflat_sc(state[off : off + n_sc])
            off += n_sc
            c00f = list(state[off : off + N_ACCUMS])
            off += N_ACCUMS
            c01f = list(state[off : off + N_ACCUMS])
            off += N_ACCUMS
            c10f = list(state[off : off + N_ACCUMS])
            off += N_ACCUMS
            c11f = list(state[off : off + N_ACCUMS])
            accs = (c00f, c01f, c10f, c11f)
            # step K_ITERS - 2 (even): loads of the next tile's step 0, gather of its block 0
            scale_gather.set_tile_voff(Gn)
            a0f, b0f, b1f, sc, accs, bufs = _one_step(
                0,
                a0f,
                b0f,
                b1f,
                sc,
                accs,
                bufs0,
                _g2s_bases(bases_n, 0),
                (_gm0_s[0], 0),
                None,
            )
            # step K_ITERS - 1 (odd): loads of the next tile's step 1, read of its block 0;
            # this tile's c00 / c01 stored over the groups after their last MFMAs (c00: 1..3,
            # c01: 2..3; the two quadrants' thunks stay concatenated, they share the LDS buffer)
            n_q3 = (
                2 * store_c.n_thunks + 2
            ) // 3  # thunks per group: two quadrants over 3 groups

            def _last_extra(g, c00, c01):
                # (compile-time python only: no mutable state -- the DSL rewriter turns
                # body-level literals into None -- and every `if` under const_expr)
                if const_expr(g == 0):
                    return ()
                t00 = store_c.thunks(c00, base_row, base_col)
                if const_expr(g == 1):
                    return t00[:n_q3]
                t01 = store_c.thunks(c01, base_row, base_col + LDS_BLOCK_N)
                if const_expr(g == 2):
                    return t00[n_q3:] + t01[: 2 * n_q3 - len(t00)]
                return t01[2 * n_q3 - len(t00) :]

            a0f, b0f, b1f, sc, accs, bufs = _one_step(
                1,
                a0f,
                b0f,
                b1f,
                sc,
                accs,
                bufs,
                _g2s_bases(bases_n, 1),
                None,
                0,
                extra=_last_extra,
            )
            new_tile_state = (
                [_R(x) for x in a0f] + [_R(x) for x in b0f] + [_R(x) for x in b1f] + _flat_sc(sc)
                + [_R(x) for x in accs[2]] + [_R(x) for x in accs[3]] + [_R(t_next), _R(t_cur)]
            )  # fmt: skip
            tstate = yield new_tile_state

        # ---- epilogue: the last tile's c10 / c11 ----
        off = n_a + 2 * n_b + n_sc
        c10_last = list(tstate[off : off + N_ACCUMS])
        off += N_ACCUMS
        c11_last = list(tstate[off : off + N_ACCUMS])
        off += N_ACCUMS
        t_last = fx.Int32(tstate[off + 1])
        til, tjl = tile_coords(t_last)
        last_row1 = til * BLOCK_M + wave_i * (N_TILES_A * 16) + LDS_BLOCK_M
        last_col0 = tjl * BLOCK_N + wave_j * (N_TILES_B * 16)
        store_c.store(c10_last, last_row1, last_col0)
        store_c.store(c11_last, last_row1, last_col0 + LDS_BLOCK_N)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        tile_off: fx.Int32,
        n_work: fx.Int32,
        stream: fx.Stream,
    ):
        """work items [tile_off, tile_off + n_work / SUB_SPLIT) of the parent grid
        ceil(M / 256) x ceil(N / 256); the whole GEMM is (0, parents * SUB_SPLIT)"""
        grid_x = _min(n_work, fx.Int32(GRID_MAX))
        kernel_gemm(
            A, B_T, C, A_scale, B_scale, c_m, c_n, tile_off, n_work,
            value_attrs={"rocdl.waves_per_eu": 1, "rocdl.flat_work_group_size": "256,256"},
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)  # fmt: skip

    launch_gemm.kernel_name = name
    launch_gemm.sub_split = SUB_SPLIT
    return launch_gemm
