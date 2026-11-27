# SPDX-License-Identifier: MIT
"""Naive grouped (contiguous-M) MXFP4 MoE GEMM for gfx1250.

Same math and the same routing contract as the production kernel
``aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py``::

    out[r, :] = dequant(a[r, :]) @ dequant(w[expert(r)]).T     (bf16 out, fp32 acc)

Each K tile of A, W and both e8m0 scales moves global -> LDS by TDM, then LDS
-> registers in the WMMA lane layout. Everything else is still stripped:

* weights in the production layouts ((16,16)-preshuffled fp4, n32k4 e8m0; see
  preshuffle_weights), activations plain row-major, activation scales grouped
  per (M tile, K tile) so each K tile is one contiguous TDM run (see
  preshuffle_act_scale; production's m16k4 layout serves the same purpose),
* m_warp x n_warp waves per workgroup (each owns one sub-tile of the output,
  every wave issues its share of each of the four TDM copies), num_buffers LDS stages with
  TDM running num_buffers-1 K tiles ahead of the WMMAs, no split-K, no
  clusters / multicast; only A's LDS rows are padded (16 B, bank conflicts),
* one V_WMMA_SCALE per (16-row M, 32/16-row N) block, two register slots so
  each K step's WMMAs overlap the next step's LDS reads (across K tiles too),
* bf16 epilogue staged in LDS and written by TDM (no bias, no activation, no
  MX requant, no EP scatter),
* workgroups in production's grouped swizzle order (GROUP_M M tiles per sweep
  over N, so the workgroups sharing an A tile land on one XCD and its L2),
* K is zero-padded on the host to a multiple of tile_k.

Activations are either MXFP4 (``a4w4``) or MXFP8 e4m3 (``a8w4``); weights are
always MXFP4. Every scale is one e8m0 per 32 K elements.

Routing (identical to production): rows are grouped by expert, each expert's
rows start on a ``tile_m`` boundary, and ``psum[e]`` is the end (exclusive) of
expert ``e``'s valid rows::

    starts[e] = sum_{i<e} round_up(rows[i], tile_m)
    psum[e]   = starts[e] + rows[e]

so the owner of an M tile is the number of experts whose ``psum`` is <= the
tile's first row, and rows in ``[psum[e], starts[e+1])`` are padding that is
neither read back nor written.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.nn.functional as F
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops
from aiter.ops.flydsl.kernels.tensor_shim import (
    _run_compiled,
    current_stream_arg,
    ptr_arg,
    ptr_buf_tensor,
)
from aiter.ops.shuffle import shuffle_scale_n32k4, shuffle_weight
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

WAVE = 32  # gfx1250 is wave32
WMMA_K = 128  # K of one V_WMMA_SCALE = 4 e8m0 blocks of 32


def ceildiv(a, b):
    return (a + b - 1) // b


def _concat(parts, width=4):
    """[v0, v1, v2, v3] -> v0 ++ v1 ++ v2 ++ v3 (equal-width Vecs)."""
    while len(parts) > 1:
        parts = [
            parts[i].shuffle(parts[i + 1], list(range(2 * width)))
            for i in range(0, len(parts), 2)
        ]
        width *= 2
    return parts[0]


@functools.cache
def build_naive_grouped_gemm(
    *, N, K, n_experts, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, a_is_fp4
):
    """Return the ``@flyc.jit`` launcher for one compile-time configuration.

    ``K`` must already be a multiple of ``tile_k`` (the host wrapper pads it).
    """
    assert tile_k % WMMA_K == 0, f"tile_k={tile_k} must be a multiple of {WMMA_K}"
    assert K % tile_k == 0, f"K={K} must be padded to a multiple of tile_k={tile_k}"

    # ---- one V_WMMA_SCALE --------------------------------------------------
    # The weight tile is WMMA operand A and the activation tile is operand B,
    # so the instruction computes a (weight rows) x (activation rows) block.
    #   a4w4: 32 x 16 x 128, fp4 weights x fp4 activations
    #   a8w4: 16 x 16 x 128, fp4 weights x fp8 activations
    W_ROWS = 32 if a_is_fp4 else 16  # N rows per WMMA
    A_ROWS = 16  # M rows per WMMA
    ACC_LEN = W_ROWS * A_ROWS // WAVE  # fp32 results per lane: 16 or 8
    act_elem = fx.Float4E2M1FN if a_is_fp4 else fx.Float8E4M3FN

    # ---- waves: an m_warp x n_warp grid, each owning a WARP_M x WARP_N block --
    NUM_WAVES = m_warp * n_warp
    BLOCK = NUM_WAVES * WAVE
    WARP_M, WARP_N = tile_m // m_warp, tile_n // n_warp
    assert WARP_M % A_ROWS == 0, f"tile_m/m_warp={WARP_M} must be a multiple of {A_ROWS}"
    assert WARP_N % W_ROWS == 0, f"tile_n/n_warp={WARP_N} must be a multiple of {W_ROWS}"
    assert N % tile_n == 0, f"N={N} must be a multiple of tile_n={tile_n}"
    WM_REP = WARP_M // A_ROWS  # WMMAs per wave along M
    WN_REP = WARP_N // W_ROWS  # WMMAs per wave along N

    # ---- plain row-major byte layouts ---------------------------------------
    K_TILES = K // tile_k  # TDM loads per operand
    K_STEPS = tile_k // WMMA_K  # WMMA K steps per K tile
    A_ROW_BYTES = K // 2 if a_is_fp4 else K
    W_ROW_BYTES = K // 2  # weights are always fp4
    SC_ROW_BYTES = K // 32  # one e8m0 per 32 K
    A_TILE_BYTES = A_ROW_BYTES // K_TILES  # one row inside one K tile
    W_TILE_BYTES = W_ROW_BYTES // K_TILES
    SC_TILE_BYTES = SC_ROW_BYTES // K_TILES  # one dword per WMMA K step
    AS_BLOCK_BYTES = tile_m * SC_TILE_BYTES  # A scales of one (M tile, K tile)
    AS_ROW_BYTES = K_TILES * AS_BLOCK_BYTES  # ... of one M tile, every K tile
    A_STEP_BYTES = A_TILE_BYTES // K_STEPS  # one row inside one WMMA K step
    W_STEP_BYTES = W_TILE_BYTES // K_STEPS
    A_DWORDS = A_ROWS * A_STEP_BYTES // (4 * WAVE)  # 8 (fp4) or 16 (fp8)
    W_DWORDS = W_ROWS * W_STEP_BYTES // (4 * WAVE)  # 16 (a4w4) or 8 (a8w4)
    # (16,16)-preshuffled weights: 16-row groups, one group's K tile is one run of
    # 16 * W_TILE_BYTES bytes. n32k4 scales: 32-row groups, 128 B per WMMA K step.
    assert tile_n % 32 == 0, f"tile_n={tile_n}: n32k4 scales come in 32-row groups"
    WG_ROW_BYTES, WG_TILE_BYTES = 16 * W_ROW_BYTES, 16 * W_TILE_BYTES
    SB_ROW_BYTES, SB_TILE_BYTES = 128 * (K // WMMA_K), 128 * K_STEPS

    # ---- LDS: NB stages, each one K tile of every operand, row-major ----------
    # A rows get 16 B of slack: with a power-of-two pitch the 16 rows one
    # ds_load_b128 reads start on only 1-4 distinct banks (same as LDS_PAD_A).
    A_LDS_PITCH = A_TILE_BYTES + 16
    A_LDS = 0
    W_LDS = A_LDS + tile_m * A_LDS_PITCH
    AS_LDS = W_LDS + tile_n * W_TILE_BYTES
    WS_LDS = AS_LDS + tile_m * SC_TILE_BYTES
    STAGE_BYTES = WS_LDS + tile_n * SC_TILE_BYTES
    # Step 0 of tile kt+1 is read while tile kt still owns its stage: >= 2 stages.
    NB = min(max(num_buffers, 2), K_TILES)
    # bf16 C tile, reusing the stages; +16 columns spread the ds_store_b128 banks.
    C_PITCH = tile_n + 16
    LDS_BYTES = max(NB * STAGE_BYTES, tile_m * C_PITCH * 2)
    # TDM copies per wave per K tile (its share of A, W and both scales);
    # s_wait_tensorcnt counts this wave's own.
    COPIES = 4

    fmt = "a4w4" if a_is_fp4 else "a8w4"
    name = (
        f"naive_grouped_gemm_tdm_{fmt}_t{tile_m}x{tile_n}x{tile_k}_w{m_warp}x{n_warp}"
        f"_b{NB}_N{N}_K{K}_E{n_experts}"
    )
    # Workgroup order (production's TILES_PER_GROUP swizzle): linear ids walk GROUP_M
    # M tiles for one N tile, then the same M tiles for the next one. The N_TILES
    # workgroups that read one A tile get ids j, j + GROUP_M, ...: the same id % 8,
    # i.e. the same XCD and L2 (workgroup w runs on XCD w % 8), and close in time.
    GROUP_M = 16
    N_TILES = N // tile_n

    @flyc.kernel(name=name, known_block_size=[BLOCK, 1, 1])
    def kernel(
        c: fx.Pointer,  # [M, N] bf16
        a: fx.Pointer,  # [M, A_ROW_BYTES]
        w: fx.Pointer,  # [E, N, K/2]
        a_scale: fx.Pointer,  # [M/tile_m, K_TILES, tile_m, SC_TILE_BYTES] e8m0
        w_scale: fx.Pointer,  # [E, N, K/32] e8m0
        psum: fx.Pointer,  # [E] int32
        m_tiles: fx.Int32,  # M tiles in the launched grid
    ):
        tid = fx.Int32(fx.thread_idx.x)
        lane = tid % WAVE
        row16 = lane % 16  # operand row (and output row) owned by this lane
        khalf = lane // 16  # 16-byte half of every 32-byte K group it owns
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // WAVE))  # wave-uniform
        wm_base = (wave // n_warp) * WARP_M  # this wave's block inside the tile
        wn_base = (wave % n_warp) * WARP_N
        bid = fx.Int32(fx.block_idx.x)
        group = bid // (GROUP_M * N_TILES)
        in_group = bid - group * (GROUP_M * N_TILES)
        rest = m_tiles - group * GROUP_M  # the last group may hold fewer M tiles
        group_tiles = (rest < GROUP_M).select(rest, GROUP_M)
        m_tile = group * GROUP_M + in_group % group_tiles
        blk_m = m_tile * tile_m
        blk_n = (in_group // group_tiles) * tile_n

        # ---- 1. which expert owns rows [blk_m, blk_m + tile_m)? ------------
        # Experts start on tile_m boundaries, so the owner is the number of
        # experts that end at or before blk_m. Each lane counts a strided slice
        # of psum, then the wave sums the counts.
        psum_t = ptr_buf_tensor(psum, fx.Int32)
        hits = fx.Int32(0)
        for i in range_constexpr(ceildiv(n_experts, WAVE)):
            e = lane + i * WAVE
            in_range = e < n_experts
            end_e = psum_t[in_range.select(e, n_experts - 1)]
            hits = hits + (in_range & (end_e <= blk_m)).select(1, 0)
        for shift in range_constexpr(5):
            hits = hits + hits.shuffle_xor(fx.Int32(1 << shift), fx.Int32(WAVE))
        expert = fx.Int32(rocdl.readfirstlane(T.i32, hits))
        expert = (expert < n_experts).select(expert, n_experts - 1)
        rows_end = psum_t[expert]  # first row past this expert's valid rows
        has_work = blk_m < psum_t[n_experts - 1]  # later tiles are all padding

        # ---- 2. TDM: K tile kt of `rows` row-major rows -> LDS ---------------
        lds = fx.SharedAllocator(static=True).allocate(LDS_BYTES)._ptr
        lds_i8 = fx.recast_iter(
            fx.PointerType.get(fx.Int8.ir_type, fx.AddressSpace.Shared, 16), lds
        )

        def tdm_k_tile(src, first_row, rows, row_bytes, tile_bytes, lds_offset, lds_pitch=None):
            """One TDM descriptor; copy(kt, stage) moves this wave's share of K tile
            kt into a stage (the atom splits the box over all NUM_WAVES waves).

            A wider lds_pitch leaves slack after every row; the atom turns it into
            the descriptor's pad interval / amount.
            """
            lds_pitch = lds_pitch or tile_bytes
            g_ptr = fx.recast_iter(fx.Int8, src) + first_row * row_bytes
            g = fx.Tensor(
                fx.make_view(g_ptr, fx.make_layout((rows, row_bytes), (row_bytes, 1)))
            )
            box = (rows, tile_bytes)
            atom, coord = fx.rocdl.cdna5.make_tiled_tdm_atom(
                fx.rocdl.TensorLoad(), g, fx.make_layout(box, (lds_pitch, 1)), box,
                num_warps=NUM_WAVES,
            )
            s = fx.Tensor(
                fx.make_view(
                    lds_i8 + lds_offset,
                    fx.make_layout((box, NB), ((lds_pitch, 1), STAGE_BYTES)),
                )
            )
            g_tiles = fx.zipped_divide(coord, box)[None, (0, None)]  # (box, K_TILES)
            s_part, g_part = fx.rocdl.cdna5.tdm_partition(
                atom, wave, fx.make_layout(NUM_WAVES, 1), s, g_tiles
            )
            return lambda kt, stage: fx.copy(atom, g_part[None, kt], s_part[None, stage])

        m_row0 = fx.Int64(blk_m)
        w_grp0 = fx.Int64(expert) * (N // 16) + fx.Int64(blk_n // 16)  # 16-row groups
        ws_grp0 = fx.Int64(expert) * (N // 32) + fx.Int64(blk_n // 32)  # 32-row groups
        tdm_loads = [
            tdm_k_tile(a, m_row0, tile_m, A_ROW_BYTES, A_TILE_BYTES, A_LDS, A_LDS_PITCH),
            tdm_k_tile(w, w_grp0, tile_n // 16, WG_ROW_BYTES, WG_TILE_BYTES, W_LDS),
            # one row per M tile; K tile kt is its kt-th contiguous block
            tdm_k_tile(a_scale, fx.Int64(m_tile), 1, AS_ROW_BYTES, AS_BLOCK_BYTES, AS_LDS),
            tdm_k_tile(w_scale, ws_grp0, tile_n // 32, SB_ROW_BYTES, SB_TILE_BYTES, WS_LDS),
        ]

        # ---- 3. register fragments, the MMA atom and the LDS readers ---------
        acc = [
            [fx.make_rmem_tensor(ACC_LEN, fx.Float32) for _ in range_constexpr(WN_REP)]
            for _ in range_constexpr(WM_REP)
        ]

        def new_slot():  # registers for one K step: A, W fragments and their scales
            def rmem(n, count):
                return [fx.make_rmem_tensor(n, fx.Int32) for _ in range_constexpr(count)]

            return rmem(A_DWORDS, WM_REP), rmem(W_DWORDS, WN_REP), rmem(1, WM_REP), rmem(1, WN_REP)

        slots = [new_slot(), new_slot()]  # K step ks lives in slots[ks % 2]
        mma = fx.make_mma_atom(
            fx.rocdl.WMMAScale(
                W_ROWS, A_ROWS, WMMA_K, fx.Float4E2M1FN, act_elem, fx.Float32
            )
        )
        lds_base = fx.Int32(fx.ptrtoint(lds))
        ld32, _ = make_lds_copy_ops(32)
        ld128, st128 = make_lds_copy_ops(128)

        # WMMA operand register layout (both operands, per 16 rows):
        #   lane l holds row l % 16 and, inside every 32-byte group of that
        #   row's K step, the 16-byte half selected by l // 16:
        #     fp4 (64 B / step):  bytes [h*16, h*16+16) and [32+h*16, 32+h*16+16)
        #     fp8 (128 B / step): bytes h*16 + 32*j + [0, 16), j = 0..3
        def load_act(base, wm, ks):
            row = wm_base + wm * A_ROWS + row16
            off = A_LDS + row * A_LDS_PITCH + ks * A_STEP_BYTES + khalf * 16
            return _concat(
                [ld128(base, off + 32 * j) for j in range_constexpr(A_STEP_BYTES // 32)]
            )

        # Preshuffled W: 16-byte K chunk c of row r in a 16-row group sits at
        # (c * 16 + r) * 16, so the lane's two chunks (c and c + 2) are 512 B apart.
        def load_weight(base, wn, ks):
            parts = []
            for half in range_constexpr(W_ROWS // 16):  # a4w4: rows 0-15, then 16-31
                group = (wn_base + wn * W_ROWS + half * 16) // 16
                chunk = ks * (W_STEP_BYTES // 16) + khalf
                off = W_LDS + group * WG_TILE_BYTES + (chunk * 16 + row16) * 16
                parts += [ld128(base, off + 512 * j) for j in range_constexpr(2)]
            return _concat(parts)

        # WMMA scale operand: one dword per lane = the 4 e8m0 scales of one row
        # for this K step (byte j scales K block j). The 32-row a4w4 weight
        # operand reads row l on lane l; a 16-row operand reads row l % 16 on
        # both lane halves, so either opsel half sees the right scales.
        def load_act_scale(base, wm, ks):
            row = wm_base + wm * A_ROWS + row16
            return ld32(base, AS_LDS + row * SC_TILE_BYTES + ks * 4)

        def load_weight_scale(base, wn, ks):  # n32k4: [32-row group][K step][row][4 B]
            row = wn_base + wn * W_ROWS + (lane if W_ROWS == 32 else row16)
            off = WS_LDS + (row // 32) * SB_TILE_BYTES + ks * 128 + (row % 32) * 4
            return ld32(base, off)

        def issue(kt, stage):  # no owner branch, so no s_nop for the TDM-then-branch hazard
            for tdm_load in tdm_loads:
                tdm_load(kt, stage)

        def wait_oldest(newer_tiles):
            """The oldest K tile in flight has landed; `newer_tiles` may still fly."""
            fx.rocdl.s_wait_tensorcnt(COPIES * newer_tiles)  # this wave's copies
            fx.barrier()  # ... and every other wave's

        def load_step(slot, stage, ks):  # LDS -> registers for one K step
            base = lds_base + stage * STAGE_BYTES
            a_f, w_f, a_s, w_s = slot
            for wm in range_constexpr(WM_REP):
                a_f[wm].store(load_act(base, wm, ks))
                a_s[wm].store(load_act_scale(base, wm, ks))
            for wn in range_constexpr(WN_REP):
                w_f[wn].store(load_weight(base, wn, ks))
                w_s[wn].store(load_weight_scale(base, wn, ks))

        def mma_step(slot):
            a_f, w_f, a_s, w_s = slot
            for wm in range_constexpr(WM_REP):
                for wn in range_constexpr(WN_REP):
                    fx.gemm(
                        mma,
                        acc[wm][wn],
                        w_f[wn],
                        a_f[wm],
                        acc[wm][wn],
                        scale_a=w_s[wn].load()[0],
                        scale_b=a_s[wm].load()[0],
                    )

        def compute(kt, next_newer):
            """Tile kt; step ks's WMMAs run while step ks+1 is read from LDS.

            The last step reads step 0 of tile kt+1 instead, after waiting for that
            tile (`next_newer` = tiles still allowed in flight behind it, None if kt
            is the last tile); that barrier also releases tile kt's stage.
            """
            for ks in range_constexpr(K_STEPS):
                mma_step(slots[ks % 2])
                if const_expr(ks + 1 < K_STEPS):
                    load_step(slots[(ks + 1) % 2], kt % NB, ks + 1)
                elif const_expr(next_newer is not None):
                    wait_oldest(next_newer)
                    load_step(slots[0], (kt + 1) % NB, 0)

        for wm in range_constexpr(WM_REP):
            for wn in range_constexpr(WN_REP):
                acc[wm][wn].store(Vec.filled(ACC_LEN, 0.0, fx.Float32))

        if has_work:
            # ---- 4. main loop: TDM runs NB-1 K tiles ahead of the WMMAs ------
            for t in range_constexpr(max(NB - 1, 1)):  # prologue (NB == 1: K_TILES == 1)
                issue(t, t)
            wait_oldest(max(NB - 2, 0))
            load_step(slots[0], 0, 0)
            for kt in range(K_TILES - (NB - 1)):  # steady state
                if const_expr(NB > 1):
                    nxt = fx.Int32(kt) + (NB - 1)
                    issue(nxt, nxt % NB)  # into the stage released last iteration
                compute(fx.Int32(kt), NB - 2 if NB > 1 else None)
            for j in range_constexpr(NB - 1):  # drain: the last NB-1 tiles
                compute(K_TILES - (NB - 1) + j, NB - 3 - j if j < NB - 2 else None)

            # ---- 5. epilogue: fp32 -> bf16 C tile in LDS, then one TDM store --
            # Accumulator layout: lane l holds output row l % 16; value v holds
            # column (v // 8) * 16 + (l // 16) * 8 + v % 8 of the WMMA block.
            rocdl.sched_barrier(0)  # keep the last tile's loads out of the epilogue
            fx.barrier()  # C overwrites the stages: every wave is done reading them
            for wm in range_constexpr(WM_REP):
                row = wm_base + wm * A_ROWS + row16
                for wn in range_constexpr(WN_REP):
                    vals = acc[wm][wn].load().to(fx.BFloat16)
                    for f in range_constexpr(ACC_LEN // 8):
                        col = wn_base + wn * W_ROWS + f * 16 + khalf * 8
                        out8 = vals.shuffle(vals, list(range(f * 8, f * 8 + 8)))
                        st128(lds_base, (row * C_PITCH + col) * 2, out8.bitcast(fx.Int32))
            fx.barrier()

            def bf16_ptr(p, space):
                return fx.recast_iter(fx.PointerType.get(fx.BFloat16.ir_type, space, 16), p)

            # The global view ends at the expert's last valid row and at column
            # tile_n, so TDM drops the padding rows and the LDS pad columns.
            g_c = fx.Tensor(
                fx.make_view(
                    bf16_ptr(c, fx.AddressSpace.Global) + (fx.Int64(blk_m) * N + fx.Int64(blk_n)),
                    fx.make_layout((rows_end - blk_m, tile_n), (N, 1)),
                )
            )
            box = (tile_m, C_PITCH)
            atom, coord = fx.rocdl.cdna5.make_tiled_tdm_atom(
                fx.rocdl.TensorStore(), g_c, fx.make_layout(box, (C_PITCH, 1)), box,
                num_warps=NUM_WAVES,
            )  # fmt: skip
            s_c = fx.Tensor(
                fx.make_view(
                    bf16_ptr(lds, fx.AddressSpace.Shared),
                    fx.make_layout((box, 1), ((C_PITCH, 1), tile_m * C_PITCH)),
                )
            )
            s_part, g_part = fx.rocdl.cdna5.tdm_partition(
                atom, wave, fx.make_layout(NUM_WAVES, 1), s_c,
                fx.zipped_divide(coord, box)[None, (0, None)],
            )  # fmt: skip
            fx.copy(atom, s_part[None, 0], g_part[None, 0])  # each wave: its rows
            fx.rocdl.s_wait_tensorcnt(0)  # the LDS tile must outlive the store

    @flyc.jit
    def launch(
        c: fx.Pointer,
        a: fx.Pointer,
        w: fx.Pointer,
        a_scale: fx.Pointer,
        w_scale: fx.Pointer,
        psum: fx.Pointer,
        m_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        m_tiles = (m_rows + tile_m - 1) // tile_m
        kernel(c, a, w, a_scale, w_scale, psum, m_tiles).launch(
            grid=(m_tiles * N_TILES, 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    return launch


def _pad_last_dim(x, size, value):
    return x if x.shape[-1] == size else F.pad(x, (0, size - x.shape[-1]), value=value)


def preshuffle_weights(w, w_scale, tile_k):
    """[E, N, K/2] fp4 and [E, N, K/32] e8m0 -> the production weight layouts.

    K is zero-padded to a multiple of tile_k first (production handles a ragged
    last K tile in the kernel); fp4 zero bytes are +0.0 and e8m0 127 is 1.0.
    """
    w, w_scale = w.view(torch.uint8), w_scale.view(torch.uint8)
    kp = ceildiv(2 * w.shape[-1], tile_k) * tile_k
    w = _pad_last_dim(w, kp // 2, 0).contiguous()
    w_scale = _pad_last_dim(w_scale, kp // 32, 127).contiguous()
    return shuffle_weight(w, layout=(16, 16)), shuffle_scale_n32k4(w_scale)


def preshuffle_act_scale(a_scale, tile_m, tile_k):
    """[M, K/32] e8m0 -> [M/tile_m, Kp/tile_k, tile_m, tile_k/32] (Kp = K rounded up).

    Row-major scales cost the A-scale TDM one tile_k/32-byte run per row; grouped
    by (M tile, K tile) they are one run per K tile. A real pipeline has the
    quantizer write this layout, so it stays outside the GEMM like the weights'.
    """
    a_scale = a_scale.view(torch.uint8)
    M = a_scale.shape[0]
    assert M % tile_m == 0, f"M={M} must be a multiple of tile_m={tile_m}"
    kp = ceildiv(32 * a_scale.shape[-1], tile_k) * tile_k
    a_scale = _pad_last_dim(a_scale, kp // 32, 127)
    blocks = a_scale.view(M // tile_m, tile_m, kp // tile_k, tile_k // 32)
    return blocks.permute(0, 2, 1, 3).contiguous()


def naive_grouped_gemm_mxfp4(
    a,
    a_scale,
    w,
    w_scale,
    psum,
    *,
    tile_m=16,
    tile_n=32,
    tile_k=128,
    m_warp=1,
    n_warp=1,
    num_buffers=2,
    out=None,
):
    """Grouped MXFP4 GEMM on the contiguous-M routed layout.

    a        [M, K/2] two fp4 per byte (low nibble first)  -> a4w4
             [M, K]   torch.float8_e4m3fn                  -> a8w4
    a_scale  [M/tile_m, Kp/tile_k, tile_m, tile_k/32] e8m0 from
             preshuffle_act_scale(a_scale, tile_m, tile_k)
    w        [E, N, Kp/2] fp4 and w_scale [E, N/32, Kp/128*128] e8m0, both from
             preshuffle_weights(..., tile_k) (Kp = K rounded up to tile_k)
    psum     [E] int32, expert e owns rows [starts[e], psum[e]); starts must be
             multiples of tile_m (build the layout with this tile_m)
    out      [M, N] bf16; rows outside every expert are left untouched

    Returns ``out``.
    """
    a_is_fp4 = a.dtype != torch.float8_e4m3fn
    a, a_scale = a.view(torch.uint8), a_scale.view(torch.uint8)
    w, w_scale = w.view(torch.uint8), w_scale.view(torch.uint8)
    n_experts, N, k_half = w.shape
    Kp = 2 * k_half
    assert Kp % tile_k == 0, "weights must come from preshuffle_weights(..., tile_k)"
    assert tuple(w_scale.shape) == (n_experts, N // 32, Kp // WMMA_K * 128)
    M = a.shape[0]
    assert tuple(a_scale.shape) == (M // tile_m, Kp // tile_k, tile_m, tile_k // 32), (
        "a_scale must come from preshuffle_act_scale(a_scale, tile_m, tile_k)"
    )
    a_bytes = Kp // 2 if a_is_fp4 else Kp
    assert a.shape[-1] <= a_bytes, f"a has {a.shape[-1]} bytes per row, Kp={Kp}"

    a = _pad_last_dim(a, a_bytes, 0).contiguous()
    a_scale = a_scale.contiguous()
    w, w_scale = w.contiguous(), w_scale.contiguous()
    psum = psum.to(torch.int32).contiguous()
    assert psum.numel() == n_experts

    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    launch = build_naive_grouped_gemm(
        N=N,
        K=Kp,
        n_experts=n_experts,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        num_buffers=num_buffers,
        a_is_fp4=a_is_fp4,
    )
    _run_compiled(
        launch,
        ptr_arg(out),
        ptr_arg(a),
        ptr_arg(w),
        ptr_arg(a_scale),
        ptr_arg(w_scale),
        ptr_arg(psum),
        M,
        current_stream_arg(),
    )
    return out
