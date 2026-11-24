# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Persistent 8-wave block-FP8 grouped GEMM over AITER's ``moe_sorting`` tiles.

The K pipeline is ``mxfp8_gemm_8wave``'s (256x256x128, 8 waves = 2 (M) x 4 (N),
the wave_m == 1 group one barrier behind, ``s_setprio`` around each MFMA burst,
128 KiB of A/B LDS double buffers, E8M0 scales staged through LDS four K-steps
at a time). What changes:

* Row tile ``m`` of the sorted list belongs to expert ``sorted_expert_ids[m]``;
  each tile rebases the weight descriptor to that expert, so the 32-bit buffer
  offsets never span the whole ``[E, N, K]`` tensor (2.4 GB for 289 experts).
* ``gather=True`` (stage 1) reads A rows straight from the token-major
  activations: each lane's row offset is ``token * K`` with the token taken from
  ``sorted_token_ids`` (padding rows name token ``T``, past the descriptor, so
  they load zeros), and the 1x128 A scales likewise from a row-major
  ``[T, K // 128]``. ``gather=False`` (stage 2) reads the sorted rows
  contiguously with a column-major ``[K // 128, M_pad]`` scale, as the dense kernel.
* The two 128-row B halves of a tile are independent weight row blocks: for
  stage 1 they are the gate rows ``[n * 128, +128)`` and the up rows
  ``[I + n * 128, +128)``, so every wave holds 32 gate and the matching 32 up
  columns; for stage 2 they are the two halves of a 256-row block.
* A persistent grid (one workgroup per CU) walks the tiles, the number of which
  it reads on device from ``num_valid_ids``. Workgroup ``w`` (on XCD ``w % 8``)
  starts at ``(w % 8) * G / 8 + w / 8`` and strides by ``G``, so each XCD runs
  32 consecutive tiles at a time: N-blocks of a row tile are adjacent, so the
  A tile and an expert's weight panel stay in that XCD's L2.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import (
    G2SLoader,
    S2RLoader,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    swizzle_128,
    wait_barrier,
)
from aiter.ops.flydsl.kernels.mxfp8_gemm_8wave import (
    MxMfma,
    barrier_if_zero,
)

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 128
N_WAVES = 8
NUM_XCDS = 8
LDS_BLOCK_M = BLOCK_M // 2
LDS_BLOCK_N = BLOCK_N // 2
N_TILES_A = BLOCK_M // 64  # 16-row MFMA tiles per wave per LDS half
N_TILES_B = BLOCK_N // 128
N_LDS_STEPS = LDS_BLOCK_M // 64  # 128-byte rows a wave copies per half, /64
A_GRP_ROWS = N_TILES_A * 16
B_GRP_ROWS = N_TILES_B * 16
B_K_STEP = 2 * 1024  # bytes per K-step inside a shuffle_weight (16, 16) 16-row strip


def _load_i32(div, idx):
    """One i32 from a buffer tensor, made wave-uniform (all lanes ask for the same one)."""
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    fx.copy(atom, fx.slice(div, (None, fx.Int32(idx))), reg)
    return fx.Int32(
        rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(reg.load()[0]).ir_value())
    )


def _load_i32_lane(div, idx):
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    fx.copy(atom, fx.slice(div, (None, fx.Int32(idx))), reg)
    return fx.Int32(reg.load()[0])


class StoreBf16Rows:
    """bf16 stores of the swapped-operand accumulators, 16 bytes per lane.

    With the weights as the MFMA's A operand, lane ``l`` of a 16x16 tile holds
    output row ``l % 16`` at columns ``4 * (l // 16) + [0, 4)``. One
    ``permlane16_swap`` between the two N-tiles of a pair gives every lane
    eight consecutive columns: even 16-lane rows q get tile 0's columns
    ``4q .. 4q + 7``, odd ones tile 1's ``4(q - 1) .. 4(q - 1) + 7`` (HK's
    ``store_acc_rows``).
    """

    def __init__(self, C, c_rows, ldc):
        self.ldc = ldc
        lane = fx.thread_idx.x % 64
        self.lane_row = lane % 16
        q = lane // 16
        self.lane_col = (q % 2 == 1).select(16 + 4 * (q - 1), 4 * q)
        gC = fx.rocdl.make_buffer_tensor(
            C, max_size=False, num_records_bytes=c_rows * ldc * 2
        )
        i32_ptr = fx.PointerType.get(
            elem_ty=fx.Int32.ir_type,
            address_space=rocdl.TargetAddressSpace.BufferDesc,
            alignment=16,
        )
        self.c_div = fx.logical_divide(
            fx.Tensor(
                fx.make_view(
                    fx.recast_iter(i32_ptr, fx.get_iter(gC)),
                    fx.make_layout(c_rows * (ldc // 2), 1),
                )
            ),
            fx.make_layout(1, 1),
        )
        self.atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        self.reg = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)

    def store(self, c_frag, base_row, base_col):
        """``c_frag[j * N_TILES_A + i]``: N-tile j (of 2), row tile i."""
        i32 = fx.Int32.ir_type
        for i in range_constexpr(N_TILES_A):
            x = Vec(c_frag[i]).to(fx.BFloat16).bitcast(fx.Int32)
            y = Vec(c_frag[N_TILES_A + i]).to(fx.BFloat16).bitcast(fx.Int32)
            words = []
            for w in range_constexpr(2):
                # No fx-level cross-lane swap; this is one v_permlane16_swap_b32.
                pair = rocdl.permlane16_swap(
                    _llvm.StructType.get_literal([i32, i32]),
                    x[w].ir_value(),
                    y[w].ir_value(),
                    False,
                    False,
                )
                words.append(
                    [fx.Int32(_llvm.extractvalue(i32, pair, [h])) for h in (0, 1)]
                )
            fx.memref_store_vec(
                Vec.from_elements(
                    [words[0][0], words[1][0], words[0][1], words[1][1]], fx.Int32
                ),
                self.reg,
            )
            row = base_row + i * 16 + self.lane_row
            idx = (row * self.ldc + base_col + self.lane_col) // 2  # in i32 words
            fx.copy(self.atom, self.reg, fx.slice(self.c_div, (None, idx)))


class MoeScaleLds:
    """E8M0 A / B block scales of one tile, staged through LDS ``CHUNK`` K-steps at a time.

    The LDS image is ``ScaleRawLds``'s 1x128 / 128x128 one, so the reads are
    unchanged: A slot ``c * 256 + row`` (a byte per dword) holds column ``c`` of
    the chunk for tile row ``row``; B slots 0 / 1 hold a dword of K // 128
    scales of the half-0 / half-1 weight row block. Every operand issues two
    copies per chunk, as there, so the pipeline's vmcnt counts hold.

    ``a_offs`` (per step, per lane) and ``a_k_stride`` locate the A bytes:
    gathered row-major ``[T, K // 128]`` (``tok * k128 + c``, stride 1) or
    column-major ``[K // 128, M_pad]`` (``c * M_pad + row``, stride M_pad).
    ``b_offs`` are the two B scale rows' byte offsets (``row * k128``).
    """

    CHUNK = 4
    BYTES = 2 * 256 * 16

    def __init__(
        self, a_rsrc, a_offs, a_k_stride, b_div, b_offs, k128, wave_id, a_row, lds
    ):
        self.a_rsrc = a_rsrc
        self.a_offs = a_offs
        self.a_k_stride = a_k_stride
        self.b_div = b_div
        self.b_offs = b_offs
        self.k128 = k128
        self.wave_id = wave_id
        self.lds = lds
        lane = fx.thread_idx.x % 64
        self.atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), 32)
        self.LdsPtr_t = fx.PointerType.get(fx.Float8E4M3FN.ir_type, 2, 512)
        self.bases = [
            fx.add_offset(lds.ptr, fx.make_int_tuple((a_row + lane % 16) * 4)),
            fx.add_offset(lds.ptr, fx.make_int_tuple(4096)),
        ]

    def dma(self, chunk):
        """Stage chunk ``chunk``; the last chunk may run past K (those bytes are never read)."""
        base_i32 = fx.Int32(fx.ptrtoint(self.lds.ptr))
        for step in range_constexpr(2):
            dst_off = chunk % 2 * self.BYTES + step * 2048 + self.wave_id * 256
            lds_dst = fx.inttoptr(self.LdsPtr_t, base_i32 + fx.Int32(dst_off))
            # A byte per lane into its own dword slot: neither the gathered
            # token rows nor the column offsets c * M_pad are dword aligned.
            rocdl.buffer_load_to_lds(
                self.a_rsrc,
                fx.to_llvm_ptr(lds_dst),
                fx.Int32(self.a_offs[step]).ir_value(),
                size_bytes=1,
                soffset=(fx.Int32(chunk * self.CHUNK) * self.a_k_stride).ir_value(),
            )
        for step in range_constexpr(2):
            dst_off = chunk % 2 * self.BYTES + 4096 + step * 2048 + self.wave_id * 256
            lds_dst = fx.inttoptr(self.LdsPtr_t, base_i32 + fx.Int32(dst_off))
            dst = fx.make_view(lds_dst, fx.make_layout(1, 1))
            # The dword never runs past the row's end (K // 128 >= 4).
            fx.copy(
                self.atom,
                fx.slice(self.b_div, (None, fx.Int32(self.b_offs))),
                dst,
                soffset=fx.Int32(min(chunk * self.CHUNK, self.k128 - 4)),
            )

    def _byte(self, op, off):
        v = fx.make_view(
            fx.recast_iter(
                fx.Uint8, fx.add_offset(self.bases[op], fx.make_int_tuple(off))
            ),
            fx.make_layout(1, 1),
        )
        return fx.Int32(v.load()[0])

    def read(self, op, half_row, n_tiles, k):
        """MFMA scale operands of ``n_tiles`` tiles, ``half_row`` rows past this wave's first row."""
        slot = k // self.CHUNK % 2 * self.BYTES
        if const_expr(op == 1):
            chunk = k // self.CHUNK
            start = min(chunk * self.CHUNK, self.k128 - 4)
            v = self._byte(op, slot + half_row // 128 * 4 + (k - start))
            return [v] * n_tiles
        return [
            self._byte(op, slot + ((k % self.CHUNK) * 256 + half_row + t * 16) * 4)
            for t in range_constexpr(n_tiles)
        ]


def compile_moe_bs_gemm_8w(
    *,
    K: int,
    N: int,
    gather: bool,
    gate_up_inter: int = 0,
    waves_per_eu: int = 2,
):
    """Build the persistent grouped GEMM launcher.

    ``N``: weight rows per expert (2 * I for stage 1, the model dim for stage 2).
    ``gate_up_inter`` = I > 0 pairs gate row block ``n`` with up row block
    ``I / 128 + n`` in one tile (stage 1); 0 takes 256 consecutive rows.
    Output (probe P1): the bf16 GEMM result per sorted row, ``[M_pad, N]``,
    columns in weight-row order.
    """
    assert K % 512 == 0, "a 128x128 B scale needs K a multiple of 512 here"
    K_ITERS = K // BLOCK_K
    assert K_ITERS % 2 == 0
    k128 = K // 128
    if gate_up_inter:
        assert N == 2 * gate_up_inter and gate_up_inter % 128 == 0
        NB = gate_up_inter // 128
    else:
        assert N % BLOCK_N == 0
        NB = N // BLOCK_N

    def half_row0(nb, h):
        """First weight row (within the expert) of LDS half ``h`` of N-block ``nb``."""
        if const_expr(gate_up_inter):
            return h * gate_up_inter + nb * LDS_BLOCK_N
        return nb * BLOCK_N + h * LDS_BLOCK_N

    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K
    n_chunks = (K_ITERS + MoeScaleLds.CHUNK - 1) // MoeScaleLds.CHUNK

    _kname = (
        f"flydsl_moe_bs_8w_{'gather' if gather else 'sorted'}_n{N}_k{K}"
        f"{'_gu' if gate_up_inter else ''}_w{waves_per_eu}"
    )

    @fx.struct
    class SharedStorage:
        S_lds: fx.Array[fx.Float8E4M3FN, 2 * MoeScaleLds.BYTES, 16]
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(name=_kname, known_block_size=[512, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B: fx.Pointer,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        m_pad: fx.Int32,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        lagging = rocdl.readfirstlane(fx.Int32.ir_type, 1 - wave_m)

        a_div = fx.logical_divide(
            make_fp8_buffer_tensor(A, F8_IR_t), fx.make_layout(1, 1)
        )
        a_scale_rsrc = fx.rocdl.get_buffer_rsrc(
            fx.get_iter(fx.rocdl.make_buffer_tensor(A_scale, max_size=False))
        )
        b_scale_div = fx.logical_divide(
            make_fp8_buffer_tensor(B_scale, F8_IR_t), fx.make_layout(1, 1)
        )
        ids_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(sorted_ids, max_size=False),
            fx.make_layout(1, 1),
        )
        eids_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(sorted_expert_ids, max_size=False),
            fx.make_layout(1, 1),
        )
        nv_div = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(num_valid_ids, max_size=False),
            fx.make_layout(1, 1),
        )
        # A raw pointer: [E, N, K] passes 2**31 bytes, past a tensor argument's i32 shape.
        b_iter = fx.recast_iter(fx.Float8E4M3FN, B)

        # Lane-only parts of the LDS-DMA addresses.
        gl_off_b = compute_global_swizzle(
            lane_id, wave_id, K, N_LDS_STEPS, preshuffled=True
        )
        if const_expr(gather):
            # Row (within a 128-row half) and swizzled column of each round.
            a_rows, a_cols = [], []
            for r in range_constexpr(N_LDS_STEPS):
                row = lane_id // 8 + wave_id * 8 + r * (N_WAVES * 8)
                _, col = swizzle_128(row, (lane_id % 8) * 16)
                a_rows.append(row)
                a_cols.append(col)
        else:
            gl_off_a = compute_global_swizzle(
                lane_id, wave_id, K, N_LDS_STEPS, preshuffled=False
            )
        # A scale slot w = step * 512 + wave * 64 + lane holds column w // 256
        # of tile row w % 256.
        sc_row = (wave_id % 4) * 64 + lane_id
        sc_col = [step * 2 + wave_id // 4 for step in range_constexpr(2)]

        # Operands swapped (weights as the MFMA's A): a lane then holds four
        # consecutive output columns of one row, which pack into wide stores.
        mfma = MxMfma(N_TILES_B, N_TILES_A, paired_tiles=False)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)
        store_c = StoreBf16Rows(C, m_pad, N)

        def run_tile(tile):
            m_blk = tile // NB
            nb = tile % NB
            expert = _load_i32(eids_div, m_blk)
            row0 = m_blk * BLOCK_M

            a_cur0 = lds.A_lds_cur_0
            a_cur1 = lds.A_lds_cur_1
            a_next0 = lds.A_lds_next_0
            a_next1 = lds.A_lds_next_1
            b_cur0 = lds.B_lds_cur_0
            b_cur1 = lds.B_lds_cur_1
            b_next0 = lds.B_lds_next_0
            b_next1 = lds.B_lds_next_1

            if const_expr(gather):
                toks = [
                    [
                        _load_i32_lane(ids_div, row0 + h * LDS_BLOCK_M + a_rows[r])
                        & 0xFFFFFF
                        for r in range_constexpr(N_LDS_STEPS)
                    ]
                    for h in range_constexpr(2)
                ]
                a_g2s = [
                    G2SLoader(
                        a_div,
                        [
                            toks[h][r] * K + a_cols[r]
                            for r in range_constexpr(N_LDS_STEPS)
                        ],
                        N_LDS_STEPS,
                        F8_IR_t,
                        wave_id,
                    )
                    for h in range_constexpr(2)
                ]
                a_base = [0, 0]
                sc_tok = _load_i32_lane(ids_div, row0 + sc_row) & 0xFFFFFF
                a_sc_offs = [sc_tok * k128 + sc_col[s] for s in range_constexpr(2)]
                a_k_stride = fx.Int32(1)
            else:
                g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS, F8_IR_t, wave_id)
                a_g2s = [g2s, g2s]
                a_base = [(row0 + h * LDS_BLOCK_M) * K for h in range_constexpr(2)]
                a_sc_offs = [
                    sc_col[s] * m_pad + row0 + sc_row for s in range_constexpr(2)
                ]
                a_k_stride = m_pad

            # This expert's [N, K] weights as their own descriptor.
            b_buf = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    b_iter + fx.Int64(expert) * (N * K), fx.make_layout(N * K, 1)
                ),
                max_size=False,
                num_records_bytes=N * K,
            )
            b_g2s = G2SLoader(
                fx.logical_divide(b_buf, fx.make_layout(1, 1)),
                gl_off_b,
                N_LDS_STEPS,
                F8_IR_t,
                wave_id,
            )
            B0_off = half_row0(nb, 0) * K
            B1_off = half_row0(nb, 1) * K
            # Half h's B scale row; lane parity picks the half (slot w % 2).
            b_sc_row = expert * (N // 128) + (
                (lane_id % 2 == 0).select(half_row0(nb, 0), half_row0(nb, 1)) // 128
            )
            raw_sc = MoeScaleLds(
                a_scale_rsrc,
                a_sc_offs,
                a_k_stride,
                b_scale_div,
                b_sc_row * k128,
                k128,
                wave_id,
                wave_m * A_GRP_ROWS,
                lds.S_lds,
            )

            def a_load(h, dst, k):
                a_g2s[h].load(dst, a_base[h] + k * BLOCK_K)

            def scale_read(k, which):
                op, h = int(which[0] == "b"), int(which[1])
                half_row = h * (LDS_BLOCK_N if op else LDS_BLOCK_M)
                return raw_sc.read(op, half_row, N_TILES_B if op else N_TILES_A, k)

            def scale_chunk_dma(k):
                # As the dense kernel: chunk k // CHUNK + 1 after step k's first barrier.
                chunk = k // MoeScaleLds.CHUNK + 1
                if const_expr(
                    k % MoeScaleLds.CHUNK == 0 and k > 0 and chunk < n_chunks
                ):
                    raw_sc.dma(chunk)

            c00 = [mfma.zero_value] * (N_TILES_A * N_TILES_B)
            c01 = list(c00)
            c10 = list(c00)
            c11 = list(c00)

            for chunk in range_constexpr(min(2, n_chunks)):
                raw_sc.dma(chunk)

            b_g2s.load(b_cur0, B0_off)
            a_load(0, a_cur0, 0)
            b_g2s.load(b_cur1, B1_off)
            a_load(1, a_cur1, 0)

            # The wave_m == 1 group runs one barrier behind.
            barrier_if_zero(lagging)

            wait_barrier(2 * N_LDS_STEPS)

            b_g2s.load(b_next0, B0_off + B_K_STEP)
            a_load(0, a_next0, 1)
            b_g2s.load(b_next1, B1_off + B_K_STEP)

            wait_barrier(3 * N_LDS_STEPS)

            for k in range_constexpr(K_ITERS - 2):
                sa0 = scale_read(k, "a0")
                sb0 = scale_read(k, "b0")
                b0_frag = b_s2r.load(b_cur0, preshuffled=True)
                a0_frag = a_s2r.load(a_cur0)
                a_load(1, a_next1, k + 1)
                rocdl.s_barrier()
                scale_chunk_dma(k)

                c00 = mfma.call(b0_frag, a0_frag, c00, sb0, sa0, k_pack=0)

                sb1 = scale_read(k, "b1")
                b1_frag = b_s2r.load(b_cur1, preshuffled=True)
                b_g2s.load(b_cur0, B0_off + (k + 2) * B_K_STEP)
                rocdl.s_barrier()

                c01 = mfma.call(b1_frag, a0_frag, c01, sb1, sa0, k_pack=0)

                sa1 = scale_read(k, "a1")
                a1_frag = a_s2r.load(a_cur1)
                a_load(0, a_cur0, k + 2)
                wait_barrier(4 * N_LDS_STEPS)

                c10 = mfma.call(b0_frag, a1_frag, c10, sb0, sa1, k_pack=0)

                b_g2s.load(b_cur1, B1_off + (k + 2) * B_K_STEP)
                wait_barrier(3 * N_LDS_STEPS)

                c11 = mfma.call(b1_frag, a1_frag, c11, sb1, sa1, k_pack=0)

                a_cur0, a_next0 = a_next0, a_cur0
                a_cur1, a_next1 = a_next1, a_cur1
                b_cur0, b_next0 = b_next0, b_cur0
                b_cur1, b_next1 = b_next1, b_cur1

            k = K_ITERS - 2
            sa0 = scale_read(k, "a0")
            sb0 = scale_read(k, "b0")
            b0_frag = b_s2r.load(b_cur0, preshuffled=True)
            a0_frag = a_s2r.load(a_cur0)
            wait_barrier(0)

            c00 = mfma.call(b0_frag, a0_frag, c00, sb0, sa0, k_pack=0)

            sb1 = scale_read(k, "b1")
            b1_frag = b_s2r.load(b_cur1, preshuffled=True)
            rocdl.s_barrier()

            c01 = mfma.call(b1_frag, a0_frag, c01, sb1, sa0, k_pack=0)

            sa1 = scale_read(k, "a1")
            a1_frag = a_s2r.load(a_cur1)
            a_load(1, a_next1, K_ITERS - 1)
            rocdl.s_barrier()

            c10 = mfma.call(b0_frag, a1_frag, c10, sb0, sa1, k_pack=0)

            b0_frag = b_s2r.load(b_next0, preshuffled=True)
            rocdl.s_barrier()

            c11 = mfma.call(b1_frag, a1_frag, c11, sb1, sa1, k_pack=0)
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

            k = K_ITERS - 1
            sa0 = scale_read(k, "a0")
            sb0 = scale_read(k, "b0")
            a0_frag = a_s2r.load(a_cur0)
            wait_barrier(0)

            c00 = mfma.call(b0_frag, a0_frag, c00, sb0, sa0, k_pack=0)

            sb1 = scale_read(k, "b1")
            b1_frag = b_s2r.load(b_cur1, preshuffled=True)
            rocdl.s_barrier()

            c01 = mfma.call(b1_frag, a0_frag, c01, sb1, sa0, k_pack=0)

            sa1 = scale_read(k, "a1")
            a1_frag = a_s2r.load(a_cur1)
            # The next tile's prologue overwrites every LDS buffer once the
            # leading group passes its closing barrier, which pairs with this
            # one: the lagging group's last LDS reads must have landed.
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10 = mfma.call(b0_frag, a1_frag, c10, sb0, sa1, k_pack=0, set_prio=False)
            c11 = mfma.call(b1_frag, a1_frag, c11, sb1, sa1, k_pack=0, set_prio=False)
            rocdl.s_setprio(0)
            # Pairs the lagging group's last barrier above (see the dense kernel).
            barrier_if_zero(rocdl.readfirstlane(fx.Int32.ir_type, wave_m))

            base_row = row0 + wave_m * A_GRP_ROWS
            col0 = half_row0(nb, 0) + wave_n * B_GRP_ROWS
            col1 = half_row0(nb, 1) + wave_n * B_GRP_ROWS
            store_c.store(c00, base_row, col0)
            store_c.store(c01, base_row, col1)
            store_c.store(c10, base_row + LDS_BLOCK_M, col0)
            store_c.store(c11, base_row + LDS_BLOCK_M, col1)

        grid = fx.Int32(fx.grid_dim.x)
        bid = fx.Int32(fx.block_idx.x)
        start = (bid % NUM_XCDS) * (grid // NUM_XCDS) + bid // NUM_XCDS
        n_tiles = (_load_i32(nv_div, 0) // BLOCK_M) * NB
        for tile, state in range(start, n_tiles, grid, init=[fx.Int32(0)]):
            run_tile(fx.Int32(tile))
            _ = yield [state[0]]

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B: fx.Pointer,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        m_pad: fx.Int32,
        grid_x: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_gemm(
            A,
            B,
            C,
            A_scale,
            B_scale,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            m_pad,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": "512,512",
            },
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch
