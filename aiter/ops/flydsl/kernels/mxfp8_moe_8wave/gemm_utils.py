# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec


# ceildiv is the canonical cdiv from the shared layer; re-exported here for the
# gemm kernels that historically imported it from this module.
def ceildiv(a, b):
    return (a + b - 1) // b


def divmod(a, b):
    """Integer divmod that works on DSL values (e.g. ``Int32``).

    The builtin ``divmod`` rejects DSL scalar types, so this uses the overloaded
    ``//`` / ``%`` operators to emit the corresponding ops.
    """
    return (a // b, a % b)


def xcd_remap_pid(num_pid_m, num_pid_n, *, group_m, num_xcds=8, allow_ragged=False):
    """1D workgroup id -> ``(pid_m, pid_n)``, XCD-aware with M-grouped ordering."""
    num_cus = 32 * num_xcds
    swizzle_threshold = 4 * num_cus

    wgid = fx.block_idx.x
    num_wg = fx.Int32(num_pid_m * num_pid_n)
    simple_m, simple_n = divmod(wgid, num_pid_n)

    intra_xcd, xcd = divmod(wgid, num_xcds)
    wgid_remap = xcd * (num_wg // num_xcds) + intra_xcd
    if const_expr(allow_ragged):
        # The first num_wg % num_xcds XCDs get one extra CTA. This is a
        # bijection even for expert-padded MoE grids that are not divisible by 8.
        wgid_remap += fx.min(xcd, num_wg % num_xcds)
    num_wgid_in_group = group_m * num_pid_n
    group_id, intra_group = divmod(wgid_remap, num_wgid_in_group)
    first_pid_m = group_id * group_m
    group_size_m = fx.min(num_pid_m - first_pid_m, group_m)
    pid_n, intra_group_m = divmod(intra_group, group_size_m)
    pid_m = first_pid_m + intra_group_m

    use_simple = num_wg < swizzle_threshold
    if const_expr(not allow_ragged):
        use_simple = use_simple | (num_wg % num_xcds != 0)
    return (use_simple.select(simple_m, pid_m), use_simple.select(simple_n, pid_n))


def preshuffle_b(b_t):
    """Permute row-major ``B_T`` ``(N, K)`` for ``b_preshuffled=True``."""
    n, k = b_t.shape[-2:]
    assert n % 16 == 0 and k % 64 == 0, f"need N%16==0 and K%64==0, got N={n} K={k}"
    return b_t.reshape(n // 16, 16, k // 64, 4, 16).permute(0, 2, 3, 1, 4).contiguous()


def make_fp8_buffer_tensor(arg_i8, fp8_ir_t):
    # max_size=False with no num_records_bytes: cosize(layout) becomes a
    # runtime expression because TensorAdaptor defaults to layout-dynamic
    # memref (post #554), so the descriptor adapts to the actual tensor
    # extent and no longer bakes the first-call's shape into IR.
    t_i8 = fx.rocdl.make_buffer_tensor(arg_i8, max_size=False)
    iter_i8 = fx.get_iter(t_i8)
    f8_buf_ptr_ty = fx.PointerType.get(
        elem_ty=fp8_ir_t,
        address_space=TargetAddressSpace.BufferDesc,
        alignment=fx.PointerType(iter_i8.type).alignment,
    )
    iter_f8 = fx.recast_iter(f8_buf_ptr_ty, iter_i8)
    return fx.Tensor(fx.make_view(iter_f8, fx.get_layout(t_i8)))


def swizzle_128(row, col):
    offset = row * 128 + col
    swizzle = ((offset % (16 * 128)) >> 8) << 4
    swizzled_offset = offset ^ swizzle
    return swizzled_offset // 128, swizzled_offset % 128


def compute_global_swizzle(lane_id, wave_id, K, n_rounds, preshuffled):
    offsets = []
    n_waves = fx.block_dim.x // 64
    for round in range_constexpr(n_rounds):
        if const_expr(preshuffled):
            row = lane_id % 8 + wave_id * 8 + round * (n_waves * 8)
            col = (lane_id // 8) * 16
            offsets.append(
                (row // 16) * (K * 16)
                + (row % 16) * 16
                + (col // 64) * 1024
                + ((col % 64) // 16) * 256
                + (col % 16)
            )
        else:
            row = lane_id // 8 + wave_id * 8 + round * (n_waves * 8)
            col = (lane_id % 8) * 16
            r, c = swizzle_128(row, col)
            offsets.append(r * K + c)
    return offsets


class G2SLoader:
    def __init__(self, gl_src, gl_offsets, n_load_steps, lds_dtype, wave_id):
        self.g2lds_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        self.LdsPtr_t = fx.PointerType.get(lds_dtype, 2, 512)
        self.gl_src = gl_src
        self.gl_offsets = gl_offsets
        self.n_load_steps = n_load_steps
        self.wave_id = wave_id
        self.n_waves = fx.block_dim.x // 64

    def _lds_dst_at(self, lds_dst, step):
        step_off = self.wave_id * 1024 + step * (self.n_waves * 1024)
        base_i32 = fx.Int32(fx.ptrtoint(lds_dst.ptr))
        sum_i32 = base_i32 + fx.Int32(step_off)
        lds_ptr = fx.inttoptr(self.LdsPtr_t, sum_i32)
        return fx.make_view(lds_ptr, fx.make_layout(1, 1))

    def load(self, lds_dst, k_offset):
        for step in range_constexpr(self.n_load_steps):
            src = fx.slice(self.gl_src, (None, fx.Int32(self.gl_offsets[step])))
            dst = self._lds_dst_at(lds_dst, step)
            fx.copy(self.g2lds_atom, src, dst, soffset=fx.Int32(k_offset))

    def load_one(self, lds_dst, k_offset, step):
        src = fx.slice(self.gl_src, (None, fx.Int32(self.gl_offsets[step])))
        dst = self._lds_dst_at(lds_dst, step)
        fx.copy(self.g2lds_atom, src, dst, soffset=fx.Int32(k_offset))


def pack_i32x4_i32x8(lo, hi):
    # Pack two i32x4 as one i32x8
    return lo.shuffle(hi, list(range(8)))


class S2RLoader:
    def __init__(self, wave_idx, n_tiles):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles

    def _vec_load_16xf8(self, lds_src, offset):
        off_tup = fx.make_int_tuple(offset)
        ptr_off = fx.add_offset(lds_src.ptr, off_tup)
        i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
        view = fx.make_view(i8_iter, fx.make_layout(16, 1))
        return view.load()

    def load(self, lds_src, preshuffled=False):
        frag = []
        for i in range_constexpr(self.n_tiles):
            halves = []
            row = self.wave_idx * (self.n_tiles * 16) + i * 16 + self.lane_id % 16
            for step in range_constexpr(2):
                col = (self.lane_id // 16) * 16 + step * 64
                if const_expr(preshuffled):
                    offset = (row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128
                else:
                    row_swz, col_swz = swizzle_128(row, col)
                    offset = row_swz * 128 + col_swz
                v = self._vec_load_16xf8(lds_src, offset)
                halves.append(v.bitcast(fx.Int32))
            frag.append(pack_i32x4_i32x8(halves[0], halves[1]))
        return frag

    def load_one(self, lds_src, lds_offset):
        v = self._vec_load_16xf8(lds_src, lds_offset)
        return v.bitcast(fx.Int32)


class StoreC:
    def __init__(
        self, A_scale, B_scale, C, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b
    ):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        self.scaled = A_scale is not None
        assert (A_scale is None) == (
            B_scale is None
        ), "pass both epilogue scales or neither"
        # Exact byte counts from compile-time shape (BF16 C output, FP32 scales).
        # ``num_records_bytes`` is required when ``max_size=False`` -- see
        # ``make_buffer_tensor`` docstring for the silent-OOB rationale.
        c_nbytes = c_rows * c_cols * 2  # BFloat16 = 2 bytes
        gC = fx.rocdl.make_buffer_tensor(C, max_size=False, num_records_bytes=c_nbytes)
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        if const_expr(self.scaled):
            sa_nbytes = c_rows * 4  # Float32 row-wise scale
            sb_nbytes = c_cols * 4  # Float32 col-wise scale
            gSA = fx.rocdl.make_buffer_tensor(
                A_scale, max_size=False, num_records_bytes=sa_nbytes
            )
            gSB = fx.rocdl.make_buffer_tensor(
                B_scale, max_size=False, num_records_bytes=sb_nbytes
            )
            self.sa_div = fx.logical_divide(gSA, fx.make_layout(1, 1))
            self.sb_div = fx.logical_divide(gSB, fx.make_layout(1, 1))
            self.scale_atom_4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
            self.scale_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            self.reg_f32_4 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
            self.reg_f32_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
        self.out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        self.reg_bf16_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.BFloat16)

    def _load_scale_vec4(self, row):
        fx.copy(
            self.scale_atom_4,
            fx.slice(self.sa_div, (None, fx.Int32(row))),
            self.reg_f32_4,
        )
        return Vec(fx.memref_load_vec(self.reg_f32_4))

    def _load_scale_scalar(self, col):
        fx.copy(
            self.scale_atom_1,
            fx.slice(self.sb_div, (None, fx.Int32(col))),
            self.reg_f32_1,
        )
        return Vec(fx.memref_load_vec(self.reg_f32_1))[0]

    def _store_bf16(self, value_bf16, c_index):
        fx.memref_store_vec(Vec.filled(1, value_bf16, fx.BFloat16), self.reg_bf16_1)
        fx.copy(
            self.out_atom_1,
            self.reg_bf16_1,
            fx.slice(self.c_div, (None, fx.Int32(c_index))),
        )

    def store(self, c_frag, base_row, base_col):
        if const_expr(self.scaled):
            a_scales = [
                self._load_scale_vec4(base_row + i * 16 + (self.lane_id // 16) * 4)
                for i in range_constexpr(self.n_tiles_a)
            ]
            b_scales = [
                self._load_scale_scalar(base_col + i * 16 + self.lane_id % 16)
                for i in range_constexpr(self.n_tiles_b)
            ]
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + (self.lane_id // 16) * 4
            for tj in range_constexpr(self.n_tiles_b):
                col = base_col + tj * 16 + self.lane_id % 16
                col_valid = col < self.c_cols
                oob = fx.Int32(self.c_rows * self.c_cols)
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                for i in range_constexpr(4):
                    if const_expr(self.scaled):
                        out = (vec_f32[i] * (a_scales[ti][i] * b_scales[tj])).to(
                            fx.BFloat16
                        )
                    else:
                        out = vec_f32[i].to(fx.BFloat16)
                    c_index = (row + i) * self.c_cols + col
                    self._store_bf16(out, arith.select(col_valid, c_index, oob))


def wait_barrier(count):
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string=f"s_waitcnt vmcnt({count})\ns_barrier",
        constraints="",
        has_side_effects=True,
    )


class Mfma16x16x128:
    def __init__(self, n_tiles_a, n_tiles_b):
        self.atom = fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN)
        )
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b

    def idx(self, i, j):
        return i * self.n_tiles_b + j

    def _make_operand_frag(self, value):
        frag = fx.make_rmem_tensor(8, fx.Int32)
        frag.store(Vec(value))
        return frag

    def _make_accum_frag(self, value):
        frag = fx.make_rmem_tensor(4, fx.Float32)
        frag.store(Vec(value))
        return frag

    def _do_mma(self, a, b, c):
        a_frag = self._make_operand_frag(a)
        b_frag = self._make_operand_frag(b)
        c_frag = self._make_accum_frag(c)
        fx.gemm(self.atom, c_frag, a_frag, b_frag, c_frag)
        return c_frag.load().ir_value()

    def call(self, a, b, c, *, set_prio=True):
        assert len(a) == self.n_tiles_a
        assert len(b) == self.n_tiles_b
        assert len(c) == self.n_tiles_a * self.n_tiles_b

        a_frags = [
            self._make_operand_frag(a[idx]) for idx in range_constexpr(self.n_tiles_a)
        ]
        b_frags = [
            self._make_operand_frag(b[idx]) for idx in range_constexpr(self.n_tiles_b)
        ]
        c_frags = [
            self._make_accum_frag(c[idx])
            for idx in range_constexpr(self.n_tiles_a * self.n_tiles_b)
        ]
        if const_expr(set_prio):
            rocdl.s_setprio(1)
        for i in range_constexpr(self.n_tiles_a):
            for j in range_constexpr(self.n_tiles_b):
                cf = c_frags[self.idx(i, j)]
                fx.gemm(self.atom, cf, a_frags[i], b_frags[j], cf)
        if const_expr(set_prio):
            rocdl.s_setprio(0)
            rocdl.s_barrier()
        return [
            c_frags[idx].load().ir_value()
            for idx in range_constexpr(self.n_tiles_a * self.n_tiles_b)
        ]

    def call_one(self, a, b, c, i, j):
        assert i < self.n_tiles_a and j < self.n_tiles_b

        return self._do_mma(a[i], b[j], c[self.idx(i, j)])
