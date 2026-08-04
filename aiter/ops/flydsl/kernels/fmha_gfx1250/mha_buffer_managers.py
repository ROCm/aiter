# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""16-bit (bf16/fp16) Q/K/V staging managers for the gfx1250 MHA kernels.

Each manager owns the ``global -> LDS (async) -> VGPR (WMMA fragment)`` path for
one 16-bit-element operand (Q, K or V): the LDS swizzle, the async copy schedule
and the fragment read. They are self-contained — the only things a caller passes
in are the *configuration* it already maintains (hdim, gqa_ratio, kv block width,
number of waves) via the constructor, and the runtime ``warp_idx`` / ``lane_idx``
into the specific member functions that need them. Nothing here reads a shared
module-level tiling constant from the kernel; the WMMA/tiling facts intrinsic to
the managed layout live below as private module constants.

The ``16b`` suffix names the element width (16-bit): every swizzle here assumes a
``b128 == 8`` element chunk (``_BF16_BYTES``). An 8-bit (fp8) variant would need
its own manager family (different chunk arithmetic), hence the explicit width tag.

Contents:
  - ``QManager16b`` — Q loader (ring-buffered async stage, natural ``ds_load_b128``).
  - ``KManager16b`` — K loader (one-block stage, natural ``ds_load_b128`` B-fragment).
  - ``VManager16b`` — V loader (V-specific swizzle, transpose ``ds_load_tr16_b128``).
  - ``OManager16b`` — O writer (WMMA accumulator -> LDS reshape -> coalesced store).

Target: gfx1250 (MI400 / mi450), wave32, 8 waves per threadgroup (256 threads).
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl.expr import buffer_ops, gpu, rocdl

# ============================================================================
# Manager-intrinsic tiling constants (private — not the caller's config).
#
# These are fixed by the WMMA instruction + the swizzles implemented here, so
# they are NOT parameters. Anything the caller genuinely chooses (hdim, gqa_ratio,
# kv block width, wave count) arrives through a constructor argument instead.
# ============================================================================

# v_wmma_f32_16x16x32_bf16/f16 shape.
_WMMA_M = 16
_WMMA_K = 32
_BF16_BYTES = 2
_CHUNK_ELEMS = 8  # b128 = 8 bf16
_CHUNK_BYTES = _CHUNK_ELEMS * _BF16_BYTES  # 16

# gfx1250 is wave32; every swizzle/reshape here assumes 32 lanes per wave. FlyDSL
# only exposes the wave size compiler-side (GPUTarget.warp_size, from the arch),
# not as a trace-time Python int, so it is a named constant (== main file
# WAVE_SIZE). An fp8/CDNA (wave64) port would revisit the whole file, not just this.
_WAVE_LANES = 32

# Default 8-wave ("m16x8") threadgroup; override via the ``num_waves`` ctor arg.
_DEFAULT_NUM_WAVES = 8

# KV sequence block choices (columns of one QK GEMM tile).
_N_BLOCK_CHOICES = (32, 64, 128, 256)
_DEFAULT_N_BLOCK = 64

# K global->LDS async write tile: 8(kv) x 32(hdim) per warp call (one b128/lane).
_K_WR_TILE_KV = 8
_K_WR_TILE_HD = _WMMA_K

# V staging swizzle granularity (see VManager16b). A V block is stored as stacked
# 32(kv) x v_hdim sub-blocks; each is split into 32x32 tiles, each into 4(kv)x16(d)
# subtiles, with the subtile col index XOR-swizzled by (subtile row index & 1) to
# make the transpose load (ds_load_tr16_b128) bank-conflict-free.
_V_BLK_KV = 32  # kv rows per swizzle sub-block
_V_TILE = 32  # square tile side within a sub-block (kv and d)
_V_SUB_KV = 4  # subtile kv rows
_V_SUB_HD = 16  # subtile d cols
_V_WR_TILE_KV = 8  # async global->LDS write tile: kv rows per warp call
_V_WR_TILE_HD = _WMMA_K  # 32 d cols per write tile

# O staging (OManager16b): no padding -- an XOR swizzle on the 8-bf16 chunk index
# makes both the b128 store and b128 read bank-conflict-free (same idea as the
# K/QManager swizzle). MI400 LDS = 64 banks x 4 B, so an 8-bf16 (16 B) chunk spans
# one 4-bank group and bank_group(q, slot) = (q*G + slot) % 16 with G = v_hdim/8
# chunks per row. The store touches a fixed chunk-column across all 16 q-rows, so
# the swizzle must spread the slot over all 16 q; the read touches one full row
# (all G chunks) so any within-row bijection is already conflict-free. The swizzle
# ``slot = chunk ^ (q >> shift)``, shift = max(0, 4 - log2(G)), satisfies both:
# for v_hdim=128 (G=16) it is the full ``chunk ^ q``; the shift folds q's low bits
# already carried by the ``q*G`` term when G < 16. Verified conflict-free for
# v_hdim in {64,128,256} (see /tmp/async_probe/o_bank_check.py).


def _assert_multiple(name, val, mult):
    if isinstance(val, int):
        assert val % mult == 0, f"{name} must be a multiple of {mult}; got {val}"


# ============================================================================
# Q loader (global -> LDS async -> VGPR WMMA fragments)
# ============================================================================


class QManager16b:
    """Owns everything about Q: its LDS footprint and the global->LDS->VGPR load.

    Keeping this behind one object means the compute core just asks the manager
    how much LDS to reserve (``get_lds_size_in_byte``) and then hands the raw
    allocation base to ``load_q_to_vgpr`` — the per-warp sub-offset and the
    swizzled staging layout are entirely the manager's business. A future Q
    strategy (different tiling / dtype) is a drop-in replacement.

    Staging layout (per warp, tile-major): each of ``qk_hdim // _WMMA_K`` K-tiles
    is 16 rows x _WMMA_K cols bf16 (1024 B). Within a tile, 4x4 subtiles of
    4 rows x 8 cols; the 8-col b128 chunk index is XOR-swizzled with the 4-row
    subtile index to spread LDS banks. The waves stack their tiles contiguously.

    Config (caller-maintained) arrives via the constructor: ``qk_hdim``,
    ``gqa_ratio``, ``num_waves`` and the ring depth ``lds_tiles``.
    """

    def __init__(self, *, qk_hdim, gqa_ratio, num_waves=_DEFAULT_NUM_WAVES, lds_tiles=None):
        if qk_hdim % _WMMA_K != 0:
            raise ValueError(f"qk_hdim must be a multiple of {_WMMA_K}; got {qk_hdim}")
        self.qk_hdim = qk_hdim  # compile-time
        self.gqa_ratio = gqa_ratio  # compile-time
        self.num_waves = num_waves  # compile-time (threadgroup wave count)
        self.block_m = _WMMA_M * num_waves  # Q rows per threadgroup
        self.k_tiles = qk_hdim // _WMMA_K

        # Ring-buffer depth: how many K-tiles of a wave's Q live in LDS at once
        # (2 async b128 per tile). Sets both the LDS footprint and the inflight-
        # async budget so they can't drift. Default == k_tiles fully buffers Q, so
        # every async copy is in flight at once and none waits on a ds_load to free
        # a slot; smaller trades that overlap for less LDS (freeing it for K/V).
        if lds_tiles is None:
            lds_tiles = self.k_tiles
        if not 1 <= lds_tiles <= self.k_tiles:
            raise ValueError(
                f"lds_tiles must be in [1, {self.k_tiles}]; got {lds_tiles}"
            )
        self.lds_tiles = lds_tiles
        self._warp_stride = _WMMA_M * self.lds_tiles * _WMMA_K * _BF16_BYTES

    def get_lds_size_in_byte(self):
        """LDS bytes the caller must reserve for Q staging (all waves)."""
        return self.num_waves * self._warp_stride

    def _lds_byte(self, row, col_chunk, tile):
        """Swizzled LDS byte offset (within a warp region) for a 8-col b128 chunk."""
        sw = col_chunk ^ (row // 4)  # 4x4 subtile XOR swizzle
        return (
            tile * (_WMMA_M * _WMMA_K * _BF16_BYTES)
            + row * (_WMMA_K * _BF16_BYTES)
            + sw * _CHUNK_BYTES
        )

    def _async_load_vram_to_lds(self, q_base_i64, g_off, lds_off):
        """gfx1250 async 16B global->LDS copy. ``offset``=0: a nonzero imm shifts
        BOTH src and dst by the same bytes, which our tile/half terms can't use."""
        gptr = buffer_ops.create_llvm_ptr(q_base_i64 + fx.Int64(g_off), address_space=1)
        lds_ptr = buffer_ops.create_llvm_ptr(lds_off, address_space=3)
        rocdl_dialect.global_load_async_to_lds_b128(gptr, lds_ptr, 0, 0)

    def _load_lds_to_vgpr(self, lds_off, vec_ty):
        """ds_load ``vec_ty`` from LDS byte offset ``lds_off`` into a VGPR vector."""
        lds_ptr = buffer_ops.create_llvm_ptr(lds_off, address_space=3)
        return fx.Vector(llvm_dialect.load(vec_ty, lds_ptr))

    def load_q_to_vgpr(
        self,
        *,
        ptr_Q,
        stride_q_seq,
        stride_q_head,
        q_start,
        q_len,
        kv_head,
        block_x,  # fx.Int32: this workgroup's grid-x tile index
        warp_idx,
        lane_idx,
        ptr_lds,  # fx.Int32: base byte addr of the caller's Q allocation
        scale,  # fx.Float32: softmax scale, folded into Q (HK MLA v4 style)
    ):
        """Stage this warp's 16 x qk_hdim Q tile and return the WMMA A fragments.

        Software-pipelined ring buffer of ``lds_tiles`` slots: prime the
        first slots with async global->LDS (swizzled), then for each K-tile wait
        (graduated ``s_wait_asynccnt``), ds_load_b128 the slot -> v16 bf16 frag,
        refill the freed slot with a future tile, and pre-scale by ``scale``.
        Rows with seq >= q_len are clamped in-bounds and masked later in softmax.
        """
        lds_q_base = ptr_lds + warp_idx * self._warp_stride
        q_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(ptr_Q)))
        warp_row0 = block_x * self.block_m + warp_idx * _WMMA_M

        def _issue_async(tile, slot):
            # Coalesced global read -> per-lane swizzled LDS write of logical
            # ``tile`` into physical ring ``slot`` (2 half-loads).
            for half in fx.range_constexpr(2):
                row = lane_idx // 4 + half * 8  # row within warp [0,16)
                col_chunk = lane_idx % 4  # 8-col chunk within the 32-col tile [0,4)
                pr = warp_row0 + row
                q_head = kv_head * self.gqa_ratio + pr % self.gqa_ratio
                seq = pr // self.gqa_ratio
                safe_seq = (seq < q_len).select(seq, fx.Int32(0))  # clamp OOB
                token = q_start + safe_seq
                g_off = (
                    token * stride_q_seq
                    + q_head * stride_q_head
                    + fx.Int32(tile * _WMMA_K * _BF16_BYTES)
                    + col_chunk * _CHUNK_BYTES
                )
                self._async_load_vram_to_lds(
                    q_base_i64, g_off, lds_q_base + self._lds_byte(row, col_chunk, slot)
                )

        # Prime the ring: tiles 0..lds_tiles-1 map 1:1 onto slots 0..lds_tiles-1.
        for tile in fx.range_constexpr(self.lds_tiles):
            _issue_async(tile, tile)

        v8_ty = fx.Vector.make_type(_CHUNK_ELEMS, fx.BFloat16)
        # bf16 scale -> packed v_pk_mul_bf16 (no f32 round-trip); an fx.Float32
        # scale would widen the fragment to f32.
        scale_bf16 = scale.to(fx.BFloat16)
        q_frags = []
        for tile in fx.range_constexpr(self.k_tiles):
            # Graduated wait: tile t's writes are async ops 2t/2t+1; wait until
            # only its (and later primed/refilled) copies remain. While still
            # refilling: 2*(lds_tiles-1); after: 2*(k_tiles-t-1). Equal at t==k-n.
            if tile < self.k_tiles - self.lds_tiles:
                rocdl.s_wait_asynccnt((self.lds_tiles - 1) * 2)
            else:
                rocdl.s_wait_asynccnt((self.k_tiles - tile - 1) * 2)
            slot = tile % self.lds_tiles
            row = lane_idx % _WMMA_M
            klane = lane_idx // _WMMA_M  # 0 or 1
            lo = self._load_lds_to_vgpr(lds_q_base + self._lds_byte(row, klane, slot), v8_ty)
            hi = self._load_lds_to_vgpr(lds_q_base + self._lds_byte(row, klane + 2, slot), v8_ty)
            # Refill the just-read slot with a future tile. The compiler tracks
            # only the LDS->VGPR read dependency (dscnt before the mul), so guard
            # it manually.
            if tile < self.k_tiles - self.lds_tiles:
                rocdl.s_wait_dscnt(0)
                _issue_async(self.lds_tiles + tile, slot)
            frag = lo.shuffle(hi, list(range(16)))  # v16 bf16, concat(lo, hi)
            frag = frag * scale_bf16
            # f32-precision fallback if bf16 scale hurts quality:
            # frag = (frag.to(fx.Float32) * scale).to(fx.BFloat16)
            q_frags.append(frag)
        return q_frags


# ============================================================================
# K loader (global -> LDS async -> VGPR WMMA B-fragments)
# ============================================================================


class KManager16b:
    """Owns K's LDS staging and the global->LDS->VGPR B-fragment load.

    Unlike QManager16b there is no ring buffer here: KManager16b only reports the
    byte size of ONE ``n_block x qk_hdim`` K block (``get_lds_size_in_byte``). The
    caller reserves however many ping-pong buffers it wants and passes the chosen
    buffer base (``ptr_lds``) into every method — the manager is not bound to a
    buffer. The K block is shared by all waves (each computes S[16, n_block]).

    LDS layout mirrors QManager16b: ``(n_block/16)`` kv-subtiles x ``(qk_hdim/32)``
    hdim-units, each a 16x32 bf16 tile (1024 B) with the 4x4 XOR swizzle
    (``sw = chunk ^ row//4``). The global->LDS write API streams 8(kv)x32(hdim)
    tiles (``row_idx`` = kv, mult of 8; ``col_idx`` = hdim, mult of 32) — one b128
    per lane, each an 8-row half of a unit. The VGPR read API pulls 16x16 tiles
    (``col_idx`` mult of 16); a ``col_idx``/``col_idx+16`` pair combines into one
    16x32 WMMA B operand. Loads use ``cluster_load_async_to_lds_b128``
    (MCAST-ready; mask 0 for now).

    Config (caller-maintained) arrives via the constructor: ``qk_hdim``,
    ``n_block`` and ``num_waves``.
    """

    def __init__(self, *, qk_hdim, n_block=_DEFAULT_N_BLOCK, num_waves=_DEFAULT_NUM_WAVES):
        if qk_hdim % _WMMA_K != 0:
            raise ValueError(f"qk_hdim must be a multiple of {_WMMA_K}; got {qk_hdim}")
        if n_block not in _N_BLOCK_CHOICES:
            raise ValueError(f"n_block must be one of {_N_BLOCK_CHOICES}; got {n_block}")
        self.qk_hdim = qk_hdim  # compile-time
        self.n_block = n_block  # compile-time
        self.num_waves = num_waves  # compile-time
        self.hd_units = qk_hdim // _WMMA_K  # 32-hdim WMMA units
        # 8x32 write-tile grid (async global->LDS): rows = kv, cols = hdim.
        self.n_wr_tile_rows = n_block // _K_WR_TILE_KV
        self.n_wr_tile_cols = qk_hdim // _K_WR_TILE_HD

    def get_lds_size_in_byte(self):
        """LDS bytes for one n_block x qk_hdim K block (one ping-pong buffer)."""
        return self.n_block * self.qk_hdim * _BF16_BYTES

    def _lds_byte(self, tile_row, tile_col, row_in_tile, chunk):
        """Swizzled LDS byte offset within one K block. LDS is a grid of 16x32
        WMMA tiles; ``tile_row``/``tile_col`` index that grid (row = kv,
        col = hdim). ``row_in_tile`` (0..15) is the row inside the tile and
        ``chunk`` (0..3) the 8-hdim b128 within the 32-wide tile. Same 4x4 XOR
        swizzle as QManager16b."""
        tile_base = (
            (tile_row * self.hd_units + tile_col) * (_WMMA_M * _WMMA_K * _BF16_BYTES)
        )
        sw = chunk ^ (row_in_tile // 4)
        return (
            fx.Int32(tile_base)
            + row_in_tile * (_WMMA_K * _BF16_BYTES)
            + sw * _CHUNK_BYTES
        )

    def async_load_vram_to_lds_wr_tile(
        self,
        *,
        ptr_lds,
        k_base_i64,  # fx.Int64: ptrtoint(get_iter(ptr_K))
        stride_k_seq,
        stride_k_head,
        kv_head,
        kv_row0,  # fx.Int32: global token of this block's kv-row 0
        kv_valid,  # fx.Int32: valid kv rows in this block (only read if check_oob)
        row_idx,  # kv offset within block, mult of 8, < n_block
        col_idx,  # hdim offset, mult of 32, < qk_hdim
        lane_idx,
        check_oob=True,  # compile-time: clamp rows >= kv_valid in-bounds
    ):
        """One warp streams an 8(kv) x 32(hdim) tile into ``ptr_lds`` — one b128
        (8 bf16) per lane: lane -> (wr_row = lane//4, chunk = lane%4). ``row_idx``/
        ``col_idx`` may be Python ints or fx values (all addressing below is plain
        integer arithmetic). When ``check_oob`` is False the caller guarantees the
        whole block is in-bounds and the clamp is skipped."""
        _assert_multiple("row_idx", row_idx, _K_WR_TILE_KV)
        _assert_multiple("col_idx", col_idx, _K_WR_TILE_HD)
        wr_row = lane_idx // 4  # kv row within the 8-row write tile [0,8)
        chunk = lane_idx % 4  # which 8-hdim b128 [0,4) -> spans the 32-wide tile
        tile_row = row_idx // _WMMA_M  # which 16-kv LDS tile
        row_in_tile = (row_idx % _WMMA_M) + wr_row  # row within that tile [0,16)
        tile_col = col_idx // _WMMA_K

        kv_row = fx.Int32(row_idx) + wr_row
        if check_oob:
            kv_row = (kv_row < kv_valid).select(kv_row, fx.Int32(0))  # clamp OOB
        token = kv_row0 + kv_row
        g_off = (
            token * stride_k_seq
            + kv_head * stride_k_head
            + (fx.Int32(col_idx) + chunk * _CHUNK_ELEMS) * _BF16_BYTES
        )
        gptr = buffer_ops.create_llvm_ptr(k_base_i64 + fx.Int64(g_off), address_space=1)
        lds_ptr = buffer_ops.create_llvm_ptr(
            ptr_lds + self._lds_byte(tile_row, tile_col, row_in_tile, chunk),
            address_space=3,
        )
        rocdl.cluster_load_async_to_lds(gptr, lds_ptr, _CHUNK_BYTES)

    def async_load_vram_to_lds(
        self,
        *,
        ptr_lds,
        ptr_K,
        stride_k_seq,
        stride_k_head,
        kv_head,
        kv_row0,
        kv_valid,
        warp_idx,
        lane_idx,
        check_oob=True,
    ):
        """Load the whole ``n_block x qk_hdim`` K block into ``ptr_lds``.

        The 8x32 write-tile grid (``n_wr_tile_rows x n_wr_tile_cols``) is spread
        round-robin across the waves: warp ``w`` streams tiles ``w, w+num_waves, ...``.
        ``row_idx``/``col_idx`` are derived from the runtime warp id, so the tile
        method runs with runtime indices here (plain integer arithmetic).
        ``check_oob`` is forwarded to each tile (skip the clamp for full blocks)."""
        n_tiles = self.n_wr_tile_rows * self.n_wr_tile_cols
        if n_tiles % self.num_waves != 0:
            raise NotImplementedError(
                f"K tile grid ({n_tiles}) must be divisible by {self.num_waves} warps; "
                f"got n_block={self.n_block}, qk_hdim={self.qk_hdim}"
            )
        k_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(ptr_K)))
        for i in fx.range_constexpr(n_tiles // self.num_waves):
            tile_id = warp_idx + fx.Int32(i * self.num_waves)
            row_idx = (tile_id // self.n_wr_tile_cols) * _K_WR_TILE_KV
            col_idx = (tile_id % self.n_wr_tile_cols) * _K_WR_TILE_HD
            self.async_load_vram_to_lds_wr_tile(
                ptr_lds=ptr_lds,
                k_base_i64=k_base_i64,
                stride_k_seq=stride_k_seq,
                stride_k_head=stride_k_head,
                kv_head=kv_head,
                kv_row0=kv_row0,
                kv_valid=kv_valid,
                row_idx=row_idx,
                col_idx=col_idx,
                lane_idx=lane_idx,
                check_oob=check_oob,
            )

    # ------------------------------------------------------------------
    def load_lds_to_vgpr_tile_as_k(self, *, ptr_lds, row_idx, col_idx, lane_idx):
        """Read one 16x16 tile as a WMMA fragment via natural ``ds_load_b128``
        (row_idx=kv, col_idx=hdim; both mult of 16) -> v8 bf16 per lane.

        lane -> (row_in_tile = lane%16, col_half = lane//16); returns this lane's 8
        hdim cols [col_idx + col_half*8 .. +8) of kv-row (row_idx + lane%16). Two
        adjacent calls (col_idx, col_idx+16) shuffle into a 16x32 WMMA fragment."""
        _assert_multiple("row_idx", row_idx, _WMMA_M)
        _assert_multiple("col_idx", col_idx, _WMMA_M)
        row_in_tile = lane_idx % _WMMA_M
        col_half = lane_idx // _WMMA_M  # 0 or 1: which 8-hdim half of the 16 cols
        tile_row = row_idx // _WMMA_M
        tile_col = col_idx // _WMMA_K
        chunk_base = (col_idx % _WMMA_K) // _CHUNK_ELEMS  # 0 or 2
        chunk = fx.Int32(chunk_base) + col_half
        v8_ty = fx.Vector.make_type(_CHUNK_ELEMS, fx.BFloat16)
        lds_ptr = buffer_ops.create_llvm_ptr(
            ptr_lds + self._lds_byte(tile_row, tile_col, row_in_tile, chunk),
            address_space=3,
        )
        return fx.Vector(llvm_dialect.load(v8_ty, lds_ptr))


# ============================================================================
# V loader (global -> LDS async -> VGPR WMMA A-fragments via transpose load)
# ============================================================================


class VManager16b:
    """Owns V's LDS staging and the global->LDS->VGPR **transpose** load.

    PV computes O^T = V^T @ P^T, so V is the WMMA A-operand ``V^T[d, kv]``. Since V
    is stored ``[kv, d]`` (d contiguous) but PV contracts over kv, the LDS->VGPR read
    uses ``ds_load_tr16_b128`` (transpose) rather than K's natural ``ds_load_b128``.
    See [[ds-load-tr16-b128-behavior]]: the load is (1) a per-lane b128 fetch where
    bank conflicts live, then (2) a fixed lane-indexed 8x8 transpose crossbar.

    K's swizzle gives a 2-way conflict on that fetch, so V needs its OWN LDS layout:
    a V block (``n_block`` kv x ``v_hdim``) is stored as stacked ``_V_BLK_KV`` x v_hdim
    sub-blocks; each sub-block is tiled into ``_V_TILE`` x ``_V_TILE`` tiles, each tile
    into ``_V_SUB_KV`` x ``_V_SUB_HD`` (4x16) subtiles, and the subtile column index is
    XOR-swizzled by ``(subtile_row_index & 1)``. That makes the transpose-load fetch
    bank-conflict-free (verified analytically for every 16x16 tile offset).

    Config (caller-maintained) arrives via the constructor: ``v_hdim``,
    ``n_block`` and ``num_waves``.
    """

    def __init__(self, *, v_hdim, n_block=_DEFAULT_N_BLOCK, num_waves=_DEFAULT_NUM_WAVES):
        if v_hdim % _V_TILE != 0:
            raise ValueError(f"v_hdim must be a multiple of {_V_TILE}; got {v_hdim}")
        if n_block % _V_BLK_KV != 0:
            raise ValueError(f"n_block must be a multiple of {_V_BLK_KV}; got {n_block}")
        self.v_hdim = v_hdim  # compile-time
        self.n_block = n_block  # compile-time
        self.num_waves = num_waves  # compile-time
        # 8x32 write-tile grid (async global->LDS): rows = kv, cols = d.
        self.n_wr_tile_rows = n_block // _V_WR_TILE_KV
        self.n_wr_tile_cols = v_hdim // _V_WR_TILE_HD

    def get_lds_size_in_byte(self):
        """LDS bytes for one ``n_block`` x ``v_hdim`` V block (one ping-pong buffer)."""
        return self.n_block * self.v_hdim * _BF16_BYTES

    def _lds_byte(self, kv_row, d_col):
        """Swizzled LDS byte offset of V element ``(kv_row, d_col)`` within one block.

        Layout: sub-block ``kv_row // 32`` -> 32x32 tile ``d_col // 32`` -> 4x16
        subtile ``(ridx = (kv_row%32)//4, cidx = (d_col%32)//16)``, with the stored
        column index ``cidx ^ (ridx & 1)``. Works for Python-int or fx operands (plain
        integer arithmetic); 8 contiguous ``d_col`` land contiguously in LDS."""
        blk = kv_row // _V_BLK_KV
        r32 = kv_row % _V_BLK_KV
        c32 = d_col % _V_TILE
        tile = d_col // _V_TILE
        ridx = r32 // _V_SUB_KV
        cidx = c32 // _V_SUB_HD
        sidx = cidx ^ (ridx % 2)  # XOR swizzle (avoid fx '&': use %2)
        rloc = r32 % _V_SUB_KV
        cloc = c32 % _V_SUB_HD
        return (
            blk * (_V_BLK_KV * self.v_hdim * _BF16_BYTES)
            + tile * (_V_TILE * _V_TILE * _BF16_BYTES)
            + (ridx * 2 + sidx) * (_V_SUB_KV * _V_SUB_HD * _BF16_BYTES)
            + (rloc * _V_SUB_HD + cloc) * _BF16_BYTES
        )

    def async_load_vram_to_lds_wr_tile(
        self,
        *,
        ptr_lds,
        v_base_i64,  # fx.Int64: ptrtoint(get_iter(ptr_V))
        stride_v_seq,
        stride_v_head,
        kv_head,
        kv_row0,  # fx.Int32: global token of this block's kv-row 0
        kv_valid,  # fx.Int32: valid kv rows in this block (only read if check_oob)
        row_idx,  # kv offset within block, mult of 8, < n_block
        col_idx,  # d offset, mult of 32, < v_hdim
        lane_idx,
        check_oob=True,  # compile-time: clamp rows >= kv_valid in-bounds
    ):
        """One warp streams an 8(kv) x 32(d) tile into ``ptr_lds`` (new V swizzle) —
        one b128 (8 bf16) per lane: lane -> (wr_row = lane//4, chunk = lane%4). The
        global read is coalesced (d contiguous); the LDS write is swizzled. When
        ``check_oob`` is False the caller guarantees the block is in-bounds."""
        _assert_multiple("row_idx", row_idx, _V_WR_TILE_KV)
        _assert_multiple("col_idx", col_idx, _V_WR_TILE_HD)
        wr_row = lane_idx // 4  # kv row within the 8-row write tile [0,8)
        chunk = lane_idx % 4  # which 8-d b128 [0,4) -> spans the 32-wide tile
        kv_row = fx.Int32(row_idx) + wr_row
        d_col = fx.Int32(col_idx) + chunk * _CHUNK_ELEMS

        safe_kv = kv_row
        if check_oob:
            safe_kv = (kv_row < kv_valid).select(kv_row, fx.Int32(0))  # clamp OOB
        token = kv_row0 + safe_kv
        g_off = token * stride_v_seq + kv_head * stride_v_head + d_col * _BF16_BYTES
        gptr = buffer_ops.create_llvm_ptr(v_base_i64 + fx.Int64(g_off), address_space=1)
        lds_ptr = buffer_ops.create_llvm_ptr(
            ptr_lds + self._lds_byte(kv_row, d_col), address_space=3
        )
        rocdl.cluster_load_async_to_lds(gptr, lds_ptr, _CHUNK_BYTES)

    def async_load_vram_to_lds(
        self,
        *,
        ptr_lds,
        ptr_V,
        stride_v_seq,
        stride_v_head,
        kv_head,
        kv_row0,
        kv_valid,
        warp_idx,
        lane_idx,
        check_oob=True,
    ):
        """Load the whole ``n_block`` x ``v_hdim`` V block into ``ptr_lds``.

        The 8x32 write-tile grid is spread round-robin across the waves (warp ``w``
        streams tiles ``w, w+num_waves, ...``); ``check_oob`` forwards to each tile."""
        n_tiles = self.n_wr_tile_rows * self.n_wr_tile_cols
        if n_tiles % self.num_waves != 0:
            raise NotImplementedError(
                f"V tile grid ({n_tiles}) must be divisible by {self.num_waves} warps; "
                f"got n_block={self.n_block}, v_hdim={self.v_hdim}"
            )
        v_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(ptr_V)))
        for i in fx.range_constexpr(n_tiles // self.num_waves):
            tile_id = warp_idx + fx.Int32(i * self.num_waves)
            row_idx = (tile_id // self.n_wr_tile_cols) * _V_WR_TILE_KV
            col_idx = (tile_id % self.n_wr_tile_cols) * _V_WR_TILE_HD
            self.async_load_vram_to_lds_wr_tile(
                ptr_lds=ptr_lds,
                v_base_i64=v_base_i64,
                stride_v_seq=stride_v_seq,
                stride_v_head=stride_v_head,
                kv_head=kv_head,
                kv_row0=kv_row0,
                kv_valid=kv_valid,
                row_idx=row_idx,
                col_idx=col_idx,
                lane_idx=lane_idx,
                check_oob=check_oob,
            )

    # ------------------------------------------------------------------
    def load_lds_to_vgpr_tile_as_v(self, *, ptr_lds, kv_idx, d_idx, lane_idx):
        """Transpose-load one 16(kv) x 16(d) tile as a PV A-fragment via
        ``ds_load_tr16_b128`` (kv_idx, d_idx both mult of 16) -> v8 bf16 per lane.

        Each lane feeds one b128 fetch; the fixed 8x8 crossbar then yields:
        lane l holds ``V[kv_idx + (l//16)*8 + e, d_idx + l%16]`` for e in 0..7 (v8[e]),
        i.e. 8 consecutive kv values at a fixed d = d_idx + l%16 -> the transposed
        ``V^T`` fragment. Fetch address per lane:
        ``V[kv_idx + (l//16)*8 + l%8, d_idx + ((l//8)%2)*8]`` (see class docstring)."""
        _assert_multiple("kv_idx", kv_idx, _WMMA_M)
        _assert_multiple("d_idx", d_idx, _WMMA_M)
        fetch_kv = fx.Int32(kv_idx) + (lane_idx // 16) * 8 + lane_idx % 8
        fetch_d = fx.Int32(d_idx) + ((lane_idx // 8) % 2) * 8
        v8_ty = fx.Vector.make_type(_CHUNK_ELEMS, fx.BFloat16)
        lds_ptr = buffer_ops.create_llvm_ptr(
            ptr_lds + self._lds_byte(fetch_kv, fetch_d), address_space=3
        )
        return fx.Vector(rocdl.ds_load_tr16_b128(v8_ty, lds_ptr))


# ============================================================================
# O writer (VGPR WMMA accumulator -> LDS reshape -> coalesced global store)
# ============================================================================


class OManager16b:
    """Owns the O epilogue: fp32 WMMA accumulator -> bf16 -> global VRAM.

    PV leaves each wave's 16 x v_hdim tile in the gfx1250 accumulator layout (row
    striped across lanes, == the K/V B-frag shape): for d-tile ``k`` lane ``l``
    holds ``O[q = l%16, d = (l//16)*8 + 16*k + {0..7}]`` (8 contiguous d). That is
    not coalesced for a global store (adjacent lanes = different q rows), so O
    bounces through per-warp staging LDS: store the accumulator in-place, re-read
    it giving each lane 8 contiguous d of a fixed q, then ``buffer_store`` coalesced.

    Staging LDS (per warp): 16(q) x v_hdim bf16, row-major with the 8-bf16 chunk
    index XOR-swizzled (``slot = chunk ^ (q >> shift)``; see the _O swizzle note
    above) -> no padding, both b128 store and read bank-conflict-free. May reuse
    the K/V LDS region (epilogue only) if the caller barriers first.

    Config (caller-maintained) via the constructor: ``v_hdim``, ``gqa_ratio``,
    ``num_waves``.
    """

    def __init__(self, *, v_hdim, gqa_ratio, num_waves=_DEFAULT_NUM_WAVES):
        if v_hdim % _WMMA_M != 0:
            raise ValueError(f"v_hdim must be a multiple of {_WMMA_M}; got {v_hdim}")
        self.v_hdim = v_hdim  # compile-time
        self.gqa_ratio = gqa_ratio  # compile-time
        self.num_waves = num_waves  # compile-time
        self.block_m = _WMMA_M * num_waves  # Q/O rows per threadgroup
        self.d_tiles = v_hdim // _WMMA_M  # 16-col WMMA output tiles == frags/lane
        self._row_bytes = v_hdim * _BF16_BYTES  # unpadded LDS row (bf16)
        self._warp_stride = _WMMA_M * self._row_bytes
        self._chunks_per_row = v_hdim // _CHUNK_ELEMS  # G = b128 chunks per row
        # XOR swizzle: slot = chunk ^ (q // _sw_div), shift = max(0, 4 - log2(G)).
        shift = max(0, 4 - int(self._chunks_per_row).bit_length() + 1)
        self._sw_div = 1 << shift

    def get_lds_size_in_byte(self):
        """LDS bytes the caller must reserve for O staging (all waves)."""
        return self.num_waves * self._warp_stride

    def _lds_byte(self, q_row, d_col):
        """Swizzled LDS byte offset (within a warp region) of O(q_row, d_col)."""
        chunk = d_col // _CHUNK_ELEMS
        sw = chunk ^ (q_row // self._sw_div)
        return q_row * fx.Int32(self._row_bytes) + sw * _CHUNK_BYTES

    def store_o_to_vram(
        self,
        *,
        o_rsrc,  # buffer resource over the O tensor (buffer_ops.create_buffer_resource)
        o_base_elems,  # fx.Int32: element offset of this (batch, ...) origin (0 for thd)
        stride_o_seq,  # elements per token step
        stride_o_head,  # elements per q-head step
        q_start,  # fx.Int32: first token of this request (varlen) / 0 (batch)
        q_len,  # fx.Int32: valid query rows; rows with seq >= q_len are masked off
        kv_head,  # fx.Int32: this workgroup's kv head
        block_x,  # fx.Int32: this workgroup's grid-x tile index
        warp_idx,
        lane_idx,
        ptr_lds,  # fx.Int32: base byte addr of the caller's O staging allocation
        o_frags,  # list[d_tiles] of v8 f32 (pre-normalized) WMMA accumulators
    ):
        """Reshape this warp's 16 x v_hdim fp32 accumulator to bf16 and store it.

        ``o_frags[k]`` is this lane's v8 fp32 for d-tile ``k`` (class docstring
        layout), already softmax-normalized (O / row-sum). Rows with seq >= q_len
        are dropped via the ``buffer_store`` mask. The mask redirects drops to
        offset ``0x7FFFFFFF``, so ``o_rsrc`` MUST carry the tensor's real
        ``num_records`` (``max_size=False`` / ``num_records_bytes``) or it faults.
        """
        if len(o_frags) != self.d_tiles:
            raise ValueError(
                f"expected {self.d_tiles} O frags (v_hdim//{_WMMA_M}); got {len(o_frags)}"
            )
        lds_warp = ptr_lds + warp_idx * self._warp_stride

        # --- write: accumulator -> LDS. lane -> (q=lane%16, d_half=(lane//16)*8);
        # d-tile k adds 16*k.
        q_st = lane_idx % _WMMA_M
        d_half = (lane_idx // _WMMA_M) * _CHUNK_ELEMS  # 0 or 8
        for k in fx.range_constexpr(self.d_tiles):
            d_col = d_half + fx.Int32(k * _WMMA_M)
            bf = o_frags[k].to(fx.BFloat16)  # v8 f32 -> v8 bf16 (v_cvt_pk_bf16_f32)
            lds_ptr = buffer_ops.create_llvm_ptr(
                lds_warp + self._lds_byte(q_st, d_col), address_space=3
            )
            llvm_dialect.store(bf.ir_value(), lds_ptr, alignment=_CHUNK_BYTES)

        # Writes must retire before the re-read (per-warp region -> intra-wave
        # counter wait, no threadgroup barrier).
        rocdl.s_wait_dscnt(0)

        # --- read: LDS -> VGPR (coalesced) -> global store. Enumerate the 16 x G
        # b128 chunks q-major, 32 per round: f = round*32 + lane -> (q = f//G,
        # d = (f%G)*8). Consecutive lanes = consecutive d of a row -> coalesced.
        v8_ty = fx.Vector.make_type(_CHUNK_ELEMS, fx.BFloat16)
        base_row = block_x * self.block_m + warp_idx * _WMMA_M
        for r in fx.range_constexpr(self.d_tiles):
            f = fx.Int32(r * _WAVE_LANES) + lane_idx  # 32 chunks dealt per round
            q_out = f // self._chunks_per_row  # q-row within this warp [0,16)
            d_rd = (f % self._chunks_per_row) * _CHUNK_ELEMS
            lds_ptr = buffer_ops.create_llvm_ptr(
                lds_warp + self._lds_byte(q_out, d_rd), address_space=3
            )
            data = fx.Vector(llvm_dialect.load(v8_ty, lds_ptr))

            pr = base_row + q_out  # global packed row
            q_head = kv_head * self.gqa_ratio + pr % self.gqa_ratio
            seq = pr // self.gqa_ratio
            valid = seq < q_len
            token = q_start + seq
            off_elems = (
                o_base_elems
                + token * stride_o_seq
                + q_head * stride_o_head
                + d_rd
            )
            buffer_ops.buffer_store(
                data,
                o_rsrc,
                off_elems * _BF16_BYTES,
                mask=valid,
                offset_is_bytes=True,
            )
