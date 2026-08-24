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

# O epilogue software pipeline (OManager16b, see its docstring). 64-col transpose
# units keep each coalesced global store a full 128B row; _O_INFLIGHT_UNITS units
# stay resident in an LDS ring and overlap.
_O_COLS_PER_TILE = 64
_O_INFLIGHT_UNITS = 2
_O_DSCNT_MAX = 63  # s_wait_dscnt SIMM16[5:0]

# LDS budget the O ring fits inside (the caller's non-current K|V slot):
# 8 waves * 2 units * 2KB = 32KB.
_O_LDS_BUDGET_BYTES = 32 * 1024

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


# ---- gfx1250 Expert Scheduling Mode 2 --------------------------------------
# DEP_MODE=2 turns the HW VA_VDST/VM_VSRC issue interlocks OFF. It is enabled via
# the `amdgpu-expert-scheduling-mode` LLVM hint the kernel passes at jit time (see
# _ensure_*_kernel). With the hint on, LLVM emits the DEP_MODE=2 setreg AND inserts
# ALL the dependency covers itself (post-RA depctr waits) for the plain intrinsic
# memory ops below: the SSA-visible RAW/WAR hazards AND the LDS RAW between the async
# global->LDS load and the ds_load that reads it back. So the kernel emits nothing
# but ordinary flydsl intrinsics in BOTH modes; mode 2 is codegen-identical to mode 0
# plus the one SCHED_MODE setreg, and scale-validated 0-NaN (512..16384, causal +
# non-causal) at perf parity with mode 0.
#
# HISTORY (why this used to be ~400 lines of inline asm): an earlier port hand-wrote
# every memory op as an opaque `has_side_effects` inline-asm block with manual depctr
# covers, on the belief that LLVM could not cover mode-2 hazards. That was both wrong
# and self-defeating — the opacity HID the async-store -> ds-load LDS RAW from LLVM
# (two opaque blocks, no SSA edge, LDS unmodeled), which mis-scheduled under DEP_MODE=2
# and produced SILENT NaN at scale (55% @16384 causal). Making the ops plain intrinsics
# exposed the dependency and let LLVM order+cover it; the whole asm+cover apparatus was
# then deleted. See memory fmha-flydsl-0-3-x-migration / fmha-m16x8-sched-mode2-unsafe.
#
# This flag is imported by the kernel module, which flips the LLVM hint in lockstep.
# It no longer changes codegen in this file — it only drives the hint. False -> mode 0.
ENABLE_SCHED_MODE2 = True


# ===========================================================================
# Memory ops are emitted inline via the plain flydsl/rocdl intrinsics
# (``buffer_ops.create_llvm_ptr`` + ``llvm_dialect.load``/``store`` /
# ``rocdl.ds_load_tr16_b128`` / ``buffer_ops.buffer_store``). There are no
# wrapper helpers: under mode 2 the ``amdgpu-expert-scheduling-mode`` LLVM hint
# makes LLVM insert all DEP_MODE=2 depctr covers itself for these SSA-visible ops,
# so the same code is correct in both modes. NOTE: LDS reads (``ds_load``) MUST be
# these plain intrinsics — never opaque inline asm — or LLVM cannot see the RAW
# against the opaque async global->LDS store and mis-orders it under DEP_MODE=2
# (the historical 55% NaN @16384 causal bug). See memory fmha-flydsl-0-3-x-migration.
# ===========================================================================


def _ir(x):
    """Unwrap an fx value to its raw MLIR ir.Value (pass-through if already raw)."""
    return x.ir_value() if hasattr(x, "ir_value") else x


def _async_load_to_lds(gptrs, lds_ptrs, *, cluster):
    """Issue a BATCH of async 16B (b128) global->LDS loads. Pure issue, no address
    math: the managers' ``global_load_ptrs`` already built the pointers.

    ``gptrs`` / ``lds_ptrs`` are equal-length lists — one entry per load:
      gptrs[i]   global (address-space 1) source pointer (fx.Int32, divergent)
      lds_ptrs[i] LDS (address-space 3) destination pointer
    A scalar (non-list) pointer is accepted and treated as a 1-load batch.
    ``cluster`` selects the MCAST form (K/V) vs plain global (Q).

    Each load is the plain rocdl intrinsic — under mode 2 the LLVM expert-scheduling
    hint inserts the RAW/WAR covers itself; under mode 0 the HW issue interlocks do."""
    if not isinstance(gptrs, (list, tuple)):
        gptrs = [gptrs]
    if not isinstance(lds_ptrs, (list, tuple)):
        lds_ptrs = [lds_ptrs]
    n = len(gptrs)
    if len(lds_ptrs) != n:
        raise ValueError(f"gptrs/lds_ptrs length mismatch: {n} vs {len(lds_ptrs)}")

    for gptr, lds_ptr in zip(gptrs, lds_ptrs):
        if cluster:
            # FlyDSL 0.3.x b128 op takes (gptr, lds_ptr, offset, mask); its
            # expr.rocdl.cluster_load_async_to_lds wrapper still passes the old
            # positional order, so call the dialect op directly with a 0 mask.
            mask0 = _ir(fx.Int32(0))
            rocdl_dialect.cluster_load_async_to_lds_b128(gptr, lds_ptr, 0, mask0)
        else:
            rocdl_dialect.global_load_async_to_lds_b128(gptr, lds_ptr, 0)


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

    def _async_load_vram_to_lds(self, gptrs, lds_ptrs):
        """gfx1250 async 16B global->LDS copy — accepts a batch (equal-length pointer
        lists, or scalars for one load). Q uses the plain (non-MCAST) global form."""
        _async_load_to_lds(gptrs, lds_ptrs, cluster=False)

    def global_load_ptrs(
        self,
        *,
        ptr_Q,
        lds_q_base,  # fx.Int32: byte base of THIS warp's LDS region
        warp_row0,  # fx.Int32: global Q-row of this warp's row 0
        kv_head,
        q_start,
        q_len,
        stride_q_seq,
        stride_q_head,
        lane_idx,
    ):
        """Pointers for EVERY ``global_load_async_to_lds_b128`` of this warp's Q tile,
        ready to hand to ``_async_load_to_lds`` with no further address math.

        Returns ``(gptrs, lds_ptrs)`` — two equal-length lists, one entry per async
        b128 group: ``k_tiles`` tiles x 2 half-loads = 8 / 12 / 16 groups for qk_hdim
        128 / 192 / 256. Pure index arithmetic (no memory op) so the caller can hoist
        ALL address VALU ahead of the load burst. ``gptrs`` are global (address-space
        1) source pointers, ``lds_ptrs`` the LDS (address-space 3) destinations. Rows
        with seq >= q_len are clamped in-bounds (masked later in softmax)."""
        q_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(ptr_Q)))
        gptrs, lds_ptrs = [], []
        for tile in fx.range_constexpr(self.k_tiles):
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
                gptrs.append(
                    buffer_ops.create_llvm_ptr(q_base_i64 + fx.Int64(g_off), address_space=1)
                )
                # LDS slot wraps mod lds_tiles: tile t reuses slot t % lds_tiles.
                slot = tile % self.lds_tiles
                lds_off = lds_q_base + self._lds_byte(row, col_chunk, slot)
                lds_ptrs.append(buffer_ops.create_llvm_ptr(lds_off, address_space=3))
        return gptrs, lds_ptrs

    def ds_load_ptrs(self, *, lds_q_base, lane_idx):
        """LDS (address-space 3) pointers for EVERY ``ds_load_b128`` read of this warp's
        Q tile, ready to hand to ``llvm_dialect.load`` with no further address math.

        Returns a flat list of ``k_tiles`` x 2 (lo, hi) = 8 / 12 / 16 read pointers:
        read ``2t`` is tile ``t``'s low 8-col half, read ``2t+1`` its high half; the
        pair shuffles into the 16x32 WMMA A fragment. Pure index math (no memory
        op)."""
        row = lane_idx % _WMMA_M
        klane = lane_idx // _WMMA_M  # 0 or 1
        ptrs = []
        for tile in fx.range_constexpr(self.k_tiles):
            slot = tile % self.lds_tiles  # match global_load_ptrs ring slot
            lo = lds_q_base + self._lds_byte(row, klane, slot)
            hi = lds_q_base + self._lds_byte(row, klane + 2, slot)
            ptrs.append(buffer_ops.create_llvm_ptr(lo, address_space=3))
            ptrs.append(buffer_ops.create_llvm_ptr(hi, address_space=3))
        return ptrs

    def load_q_to_vgpr_part1(
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
    ):
        """Part 1 of the Q load: compute all global/LDS offsets (stashed as members),
        then issue the prime chunk of async global->LDS loads. A trailing
        ``sched_barrier`` pins these above the caller's SALU so that work fills the
        load's shadow. Call ``load_q_to_vgpr_part2`` after to drain + read."""
        lds_q_base = ptr_lds + warp_idx * self._warp_stride
        warp_row0 = block_x * self.block_m + warp_idx * _WMMA_M

        # All address VALU up front (2 async b128 + 2 ds_load per tile).
        gptrs, lds_wr_ptrs = self.global_load_ptrs(
            ptr_Q=ptr_Q,
            lds_q_base=lds_q_base,
            warp_row0=warp_row0,
            kv_head=kv_head,
            q_start=q_start,
            q_len=q_len,
            stride_q_seq=stride_q_seq,
            stride_q_head=stride_q_head,
            lane_idx=lane_idx,
        )
        ds_ptrs = self.ds_load_ptrs(lds_q_base=lds_q_base, lane_idx=lane_idx)

        # Prime: issue the first lds_tiles tiles (2 loads each).
        n_prime = 2 * self.lds_tiles
        self._async_load_vram_to_lds(gptrs[:n_prime], lds_wr_ptrs[:n_prime])
        rocdl.sched_barrier(0)  # pin the prime loads above the caller's SALU

        # Stash for part 2 (drain + reads, and steady-loop refills when lds_tiles<k_tiles).
        self._q_gptrs = gptrs
        self._q_lds_wr_ptrs = lds_wr_ptrs
        self._q_ds_ptrs = ds_ptrs

    def load_q_to_vgpr_part2(self, *, scale):
        """Part 2 of the Q load: drain the async loads issued in part 1 and read the
        tiles into WMMA A fragments (``scale`` folded in). A leading ``sched_barrier``
        keeps the waits/reads below the caller's SALU so it stays in the load shadow.
        Async is drained GRADUALLY (in-issue-order assumption); ``lds_tiles ==
        k_tiles`` (default) skips the steady loop = fully-resident drain-only."""
        rocdl.sched_barrier(0)  # keep waits/reads below the caller's SALU
        k_tiles = self.k_tiles
        lds_tiles = self.lds_tiles
        ds_ptrs = self._q_ds_ptrs
        v8_ty = fx.Vector.make_type(_CHUNK_ELEMS, fx.BFloat16)

        def _read_tile(tile):
            # Self-scheduled reads: the surrounding s_wait_asynccnt / s_wait_dscnt
            # pipeline (below) already orders these against the async refills.
            lo = fx.Vector(llvm_dialect.load(v8_ty, ds_ptrs[2 * tile]))
            hi = fx.Vector(llvm_dialect.load(v8_ty, ds_ptrs[2 * tile + 1]))
            return lo, hi

        def _refill(tile):
            lo = 2 * tile
            self._async_load_vram_to_lds(
                self._q_gptrs[lo:lo + 2], self._q_lds_wr_ptrs[lo:lo + 2],
            )

        scale_bf16 = scale.to(fx.BFloat16)

        q_frags = []
        # Steady loop: read+evict tile (i-lds_tiles), refill tile i into its slot.
        for i in fx.range_constexpr(lds_tiles, k_tiles):
            rocdl.s_wait_asynccnt((lds_tiles - 1) * 2)  # oldest tile's 2 loads landed
            lo, hi = _read_tile(i - lds_tiles)
            rocdl.s_wait_dscnt(0)  # slot free to overwrite
            _refill(i)
            q_frags.append(lo.shuffle(hi, list(range(16))) * scale_bf16)
        # Drain loop: read the last lds_tiles tiles, no refill; overlap lo scale w/ hi load.
        for i in fx.range_constexpr(0, lds_tiles):
            rocdl.s_wait_asynccnt((lds_tiles - 1 - i) * 2)
            lo, hi = _read_tile(k_tiles - lds_tiles + i)
            rocdl.s_wait_dscnt(1)  # lo landed (in-order LDS return)
            lo = lo * scale_bf16
            rocdl.s_wait_dscnt(0)  # hi landed
            hi = hi * scale_bf16
            q_frags.append(lo.shuffle(hi, list(range(16))))
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

    def global_load_ptrs(
        self,
        *,
        ptr_lds,  # fx.Int32: byte base of the target K ping-pong buffer
        ptr_K,
        stride_k_seq,
        stride_k_head,
        kv_head,
        kv_row0,  # fx.Int32: global token of this block's kv-row 0
        kv_valid,  # fx.Int32: valid kv rows in this block (only read if check_oob)
        warp_idx,
        lane_idx,
        check_oob=True,  # compile-time: clamp rows >= kv_valid in-bounds
    ):
        """Src+dst pointers for EVERY ``cluster_load_async_to_lds_b128`` of this warp's
        share of the ``n_block x qk_hdim`` K block, ready to hand to
        ``_async_load_to_lds`` with no further address math. Pure index arithmetic (no
        memory op) so the caller can hoist all address VALU ahead of the load burst.

        Returns ``(gptrs, lds_ptrs)`` — equal-length pointer lists, one 8(kv)x32(hdim)
        b128 per entry (length = ``n_wr_tile_rows*n_wr_tile_cols // num_waves``);
        ``gptrs`` global (address-space 1) sources, ``lds_ptrs`` LDS (address-space 3)
        destinations. The 8x32 write-tile grid is spread round-robin across the waves
        (warp ``w`` streams tiles ``w, w+num_waves, ...``); per tile lane -> (wr_row =
        lane//4, chunk = lane%4). Rows >= ``kv_valid`` are clamped in-bounds when
        ``check_oob`` (masked later in softmax)."""
        n_tiles = self.n_wr_tile_rows * self.n_wr_tile_cols
        if n_tiles % self.num_waves != 0:
            raise NotImplementedError(
                f"K tile grid ({n_tiles}) must be divisible by {self.num_waves} warps; "
                f"got n_block={self.n_block}, qk_hdim={self.qk_hdim}"
            )
        base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(ptr_K)))
        wr_row = lane_idx // 4  # kv row within the 8-row write tile [0,8)
        chunk = lane_idx % 4  # which 8-hdim b128 [0,4) -> spans the 32-wide tile
        gptrs, lds_ptrs = [], []
        for i in fx.range_constexpr(n_tiles // self.num_waves):
            tile_id = warp_idx + fx.Int32(i * self.num_waves)
            row_idx = (tile_id // self.n_wr_tile_cols) * _K_WR_TILE_KV
            col_idx = (tile_id % self.n_wr_tile_cols) * _K_WR_TILE_HD
            tile_row = row_idx // _WMMA_M  # which 16-kv LDS tile
            row_in_tile = (row_idx % _WMMA_M) + wr_row  # row within that tile [0,16)
            tile_col = col_idx // _WMMA_K
            kv_row = row_idx + wr_row
            if check_oob:
                kv_row = (kv_row < kv_valid).select(kv_row, fx.Int32(0))  # clamp OOB
            token = kv_row0 + kv_row
            g_off = (
                token * stride_k_seq
                + kv_head * stride_k_head
                + (col_idx + chunk * _CHUNK_ELEMS) * _BF16_BYTES
            )
            gptrs.append(
                buffer_ops.create_llvm_ptr(base_i64 + fx.Int64(g_off), address_space=1)
            )
            lds_off = ptr_lds + self._lds_byte(tile_row, tile_col, row_in_tile, chunk)
            lds_ptrs.append(buffer_ops.create_llvm_ptr(lds_off, address_space=3))
        return gptrs, lds_ptrs

    # ------------------------------------------------------------------
    def ds_load_ptrs(self, *, ptr_lds, lane_idx):
        """The **2** per-lane LDS (address-space 3) base pointers that ``load_k_to_reg``
        needs to reach EVERY ``ds_load_b128`` of one K block by a compile-time immediate.

        The K swizzle within a 16x32 WMMA tile depends only on ``half`` (which 8-hdim
        half) plus the lane; every other tile ``(kv, dt)`` in the block sits at the SAME
        in-tile position shifted by the lane-independent tile stride ``(kv*hd_units +
        dt)*1024`` bytes. So the whole block's ds_load addresses collapse to just 2 base
        pointers — the ``(kv=0, dt=0)`` tile for ``half=0`` and ``half=1`` — and 16
        compile-time immediates applied in ``load_k_to_reg``. Returns
        ``[base_half0, base_half1]``. This replaces the former 32-pointer list, saving
        ~15 address VGPRs/lane and the per-pointer swizzle VALU. Pure index math."""
        row_in_tile = lane_idx % _WMMA_M
        col_half = lane_idx // _WMMA_M  # 0 or 1: which 8-hdim half of the 16 cols
        bases = []
        for half in fx.range_constexpr(2):  # rep tile (kv=0, dt=0), both halves
            col_idx = half * _WMMA_M
            chunk_base = (col_idx % _WMMA_K) // _CHUNK_ELEMS  # 0 or 2
            chunk = fx.Int32(chunk_base) + col_half
            off = ptr_lds + self._lds_byte(0, 0, row_in_tile, chunk)
            bases.append(buffer_ops.create_llvm_ptr(off, address_space=3))
        return bases

    def load_k_to_reg(self, base_ptrs, lds_imm_offset=0):
        """Burst all K ``ds_load_b128`` for the resident block from the 2 bases of
        ``ds_load_ptrs`` (buffer selected by ``base_ptrs``), in ``_qk_gemm`` order
        ``[(kv, dt, half) ...]``."""
        v8_ty = fx.Vector.make_type(8, fx.BFloat16)
        tile_stride = _WMMA_M * _WMMA_K * _BF16_BYTES  # 1024: lane-independent tile step
        NKV = self.n_block // _WMMA_M
        NDT = self.qk_hdim // _WMMA_K
        out = []
        for kv in range(NKV):
            for dt in range(NDT):
                imm = (kv * self.hd_units + dt) * tile_stride + lds_imm_offset
                for half in range(2):
                    p = base_ptrs[half]
                    if imm:
                        p = buffer_ops.get_element_ptr(p, static_byte_offset=imm)
                    out.append(fx.Vector(llvm_dialect.load(v8_ty, p)))
        return out


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

    def global_load_ptrs(
        self,
        *,
        ptr_lds,
        ptr_V,
        stride_v_seq,
        stride_v_head,
        kv_head,
        kv_row0,  # fx.Int32: global token of this block's kv-row 0
        kv_valid,  # fx.Int32: valid kv rows in this block (only read if check_oob)
        warp_idx,
        lane_idx,
        check_oob=True,  # compile-time: clamp rows >= kv_valid in-bounds
    ):
        """Src+dst pointers for EVERY ``cluster_load_async_to_lds_b128`` of this warp's
        share of the ``n_block x v_hdim`` V block, ready to hand to ``_async_load_to_lds``
        with no further address math. Pure index arithmetic (no memory op) so the caller
        can hoist all address VALU ahead of the load burst.

        Returns ``(gptrs, lds_ptrs)`` — equal-length pointer lists, one 8(kv)x32(d)
        b128 per entry (length = ``n_wr_tile_rows*n_wr_tile_cols // num_waves``);
        ``gptrs`` global (address-space 1) sources, ``lds_ptrs`` LDS (address-space 3)
        destinations (new V swizzle). The 8x32 write-tile grid is spread round-robin
        across the waves (warp ``w`` streams tiles ``w, w+num_waves, ...``); per tile
        lane -> (wr_row = lane//4, chunk = lane%4). The global read is coalesced (d
        contiguous); the LDS write is swizzled. Rows >= ``kv_valid`` are clamped
        in-bounds on the GLOBAL side when ``check_oob`` (masked later in softmax); the
        LDS position uses the UNCLAMPED kv_row."""
        n_tiles = self.n_wr_tile_rows * self.n_wr_tile_cols
        if n_tiles % self.num_waves != 0:
            raise NotImplementedError(
                f"V tile grid ({n_tiles}) must be divisible by {self.num_waves} warps; "
                f"got n_block={self.n_block}, v_hdim={self.v_hdim}"
            )
        v_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(ptr_V)))
        wr_row = lane_idx // 4  # kv row within the 8-row write tile [0,8)
        chunk = lane_idx % 4  # which 8-d b128 [0,4) -> spans the 32-wide tile
        gptrs, lds_ptrs = [], []
        for i in fx.range_constexpr(n_tiles // self.num_waves):
            tile_id = warp_idx + fx.Int32(i * self.num_waves)
            row_idx = (tile_id // self.n_wr_tile_cols) * _V_WR_TILE_KV
            col_idx = (tile_id % self.n_wr_tile_cols) * _V_WR_TILE_HD
            kv_row = fx.Int32(row_idx) + wr_row
            d_col = fx.Int32(col_idx) + chunk * _CHUNK_ELEMS
            safe_kv = kv_row
            if check_oob:
                safe_kv = (kv_row < kv_valid).select(kv_row, fx.Int32(0))  # clamp OOB
            token = kv_row0 + safe_kv
            g_off = token * stride_v_seq + kv_head * stride_v_head + d_col * _BF16_BYTES
            gptrs.append(
                buffer_ops.create_llvm_ptr(v_base_i64 + fx.Int64(g_off), address_space=1)
            )
            # LDS position uses the UNCLAMPED kv_row (global read uses clamped safe_kv).
            lds_off = ptr_lds + self._lds_byte(kv_row, d_col)
            lds_ptrs.append(buffer_ops.create_llvm_ptr(lds_off, address_space=3))
        return gptrs, lds_ptrs

    # ------------------------------------------------------------------
    def ds_load_ptrs(self, *, ptr_lds, lane_idx):
        """The **2** per-lane LDS (address-space 3) transpose-load base pointers that
        ``load_v_to_reg`` needs to reach EVERY ``ds_load_tr16_b128`` of one V block by a
        compile-time immediate.

        Unlike K (2 bases split by ``half``), the V transpose swizzle
        (``cidx ^ (ridx&1)``) makes the in-tile position depend on the output d-tile's
        PARITY: keys with even ``dt`` share one lane-relative position and odd ``dt``
        another, while ``kt``, ``half`` and even/odd-``dt`` steps are all lane-independent
        byte shifts (verified exact for all 32 lanes x all keys). So the block's transpose
        addresses collapse to 2 base pointers — the ``(kt=0, half=0)`` tile for ``dt=0``
        (even) and ``dt=1`` (odd) — plus 16 compile-time immediates applied in
        ``load_v_to_reg``. Returns ``[base_dt_even, base_dt_odd]``. This replaces the
        former 32-pointer list, saving ~15 address VGPRs/lane. Per-lane b128 fetch
        (fixed 8x8 crossbar): ``V[kv_idx + (l//16)*8 + l%8, d_idx + ((l//8)%2)*8]``."""
        bases = []
        for dp in fx.range_constexpr(2):  # rep tile (dt=dp, kt=0, half=0): dt parity
            d_idx = dp * _WMMA_M
            fetch_kv = (lane_idx // 16) * 8 + lane_idx % 8  # kv_idx == 0 (kt=0, half=0)
            fetch_d = fx.Int32(d_idx) + ((lane_idx // 8) % 2) * 8
            addr = ptr_lds + self._lds_byte(fetch_kv, fetch_d)
            bases.append(buffer_ops.create_llvm_ptr(addr, address_space=3))
        return bases

    def load_v_to_reg(self, base_ptrs, lds_imm_offset=0):
        """Burst all V ``ds_load_tr16_b128`` for the resident block from the 2 bases of
        ``ds_load_ptrs`` (buffer selected by ``base_ptrs``), in ``_pv_gemm`` order
        ``[(dt, kt, half) ...]``."""
        v8_ty = fx.Vector.make_type(8, fx.BFloat16)
        d_tiles = self.v_hdim // _WMMA_M
        nkt = self.n_block // _WMMA_K
        out = []
        for dt in range(d_tiles):
            rep_off = self._lds_byte(0, (dt % 2) * _WMMA_M)
            for kt in range(nkt):
                for half in range(2):
                    key_off = self._lds_byte(kt * _WMMA_K + half * _WMMA_M, dt * _WMMA_M)
                    imm = (key_off - rep_off) + lds_imm_offset
                    p = base_ptrs[dt % 2]
                    if imm:
                        p = buffer_ops.get_element_ptr(p, static_byte_offset=imm)
                    out.append(fx.Vector(rocdl.ds_load_tr16_b128(v8_ty, p)))
        return out


# ============================================================================
# O writer (VGPR WMMA accumulator -> LDS reshape -> coalesced global store)
# ============================================================================


class OManager16b:
    """Owns the O epilogue: fp32 WMMA accumulator -> bf16 -> global VRAM.

    PV leaves each wave's 16 x v_hdim tile in the accumulator layout: for d-tile
    ``k`` lane ``l`` holds ``O[q = l%16, d = (l//16)*8 + 16*k + {0..7}]``. Adjacent
    lanes hold different q rows, so O is transposed through per-warp staging LDS --
    store accumulator-indexed, re-read giving each lane 8 contiguous d of one q,
    then ``buffer_store`` coalesced.

    The 16 x v_hdim warp tile splits into ``n_units = v_hdim / cols_per_tile``
    self-contained transpose units. Staging LDS is an ``inflight_units``-deep ring
    of ``cols_per_tile``-wide slots, each 16(q) x cols_per_tile bf16, row-major with
    the 8-bf16 chunk XOR-swizzled (``slot = chunk ^ (q >> shift)``) -> no padding,
    b128 store and read bank-conflict-free. The units software-pipeline: unit u+1's
    ds_stores are issued ahead of unit u's ds_load re-read + coalesced buffer_store,
    hiding the LDS round-trip. ``cols_per_tile=64`` keeps each global store a full
    128B row. The caller points ``ptr_lds`` at the non-current K|V slot (holding a
    dead prefetch no wave reads), so no threadgroup barrier is needed.

    Ordering uses ``s_wait_dscnt``. DSCNT is a single in-order 6-bit counter shared
    by ds_store and ds_load, so every DS op's global issue index is tracked at
    compile time and a dependency is awaited via ``clamp(issued - 1 - gidx, 0, 63)``.

    Constructor config: ``v_hdim``, ``gqa_ratio``, ``num_waves``, ``cols_per_tile``,
    ``inflight_units``, ``lds_budget_bytes``.
    """

    def __init__(
        self,
        *,
        v_hdim,
        gqa_ratio,
        num_waves=_DEFAULT_NUM_WAVES,
        cols_per_tile=_O_COLS_PER_TILE,
        inflight_units=_O_INFLIGHT_UNITS,
        lds_budget_bytes=_O_LDS_BUDGET_BYTES,
    ):
        if v_hdim % _WMMA_M != 0:
            raise ValueError(f"v_hdim must be a multiple of {_WMMA_M}; got {v_hdim}")
        self.v_hdim = v_hdim
        self.gqa_ratio = gqa_ratio
        self.num_waves = num_waves
        self.block_m = _WMMA_M * num_waves  # Q/O rows per threadgroup
        self.d_tiles = v_hdim // _WMMA_M  # WMMA output tiles == frags/lane

        cpt = min(v_hdim, cols_per_tile)
        cpt = (cpt // _WMMA_M) * _WMMA_M  # 16-col aligned
        if cpt < _WMMA_M:
            raise ValueError(f"cols_per_tile={cols_per_tile} too small")
        if v_hdim % cpt != 0:
            raise NotImplementedError(
                f"v_hdim={v_hdim} not a multiple of cols_per_tile={cpt}"
            )
        self.cols_per_tile = cpt
        self.n_units = v_hdim // cpt
        self.tiles_per_unit = cpt // _WMMA_M  # ds_store ops per unit == ds_load ops
        self.inflight_units = min(inflight_units, self.n_units)

        # LDS geometry, per unit slot; the ring holds inflight_units slots.
        self._row_bytes = cpt * _BF16_BYTES
        self._slot_stride = _WMMA_M * self._row_bytes
        self._warp_stride = self.inflight_units * self._slot_stride
        self._chunks_per_row = cpt // _CHUNK_ELEMS  # G, b128 chunks per slot row
        # XOR swizzle: slot = chunk ^ (q // _sw_div), shift = max(0, 4 - log2(G)).
        shift = max(0, 4 - int(self._chunks_per_row).bit_length() + 1)
        self._sw_div = 1 << shift

        total = num_waves * self._warp_stride
        if total > lds_budget_bytes:
            raise ValueError(
                f"O ring {total}B (waves={num_waves} x inflight={self.inflight_units} x "
                f"slot={self._slot_stride}B) exceeds budget {lds_budget_bytes}B"
            )

    def get_lds_size_in_byte(self):
        """LDS bytes the caller must reserve for O staging (all waves, whole ring)."""
        return self.num_waves * self._warp_stride

    def _lds_byte(self, slot_idx, q_row, d_col):
        """Swizzled LDS byte offset (within a warp region) of ring-slot ``slot_idx``'s
        O(q_row, d_col) (``d_col`` is slot-local, in [0, cols_per_tile))."""
        chunk = d_col // _CHUNK_ELEMS
        sw = chunk ^ (q_row // self._sw_div)
        return slot_idx * self._slot_stride + q_row * fx.Int32(self._row_bytes) + sw * _CHUNK_BYTES

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

        ``o_frags[k]`` is this lane's v8 fp32 for d-tile ``k``, already normalized
        (O / row-sum). Rows with seq >= q_len are dropped via the buffer_store mask,
        which redirects to offset ``0x7FFFFFFF`` -- so ``o_rsrc`` MUST carry the real
        ``num_records`` (``max_size=False``) or it faults.
        """
        if len(o_frags) != self.d_tiles:
            raise ValueError(
                f"expected {self.d_tiles} O frags (v_hdim//{_WMMA_M}); got {len(o_frags)}"
            )
        lds_warp = ptr_lds + warp_idx * self._warp_stride
        q_st = lane_idx % _WMMA_M
        d_half = (lane_idx // _WMMA_M) * _CHUNK_ELEMS  # 0 or 8
        v8_ty = fx.Vector.make_type(_CHUNK_ELEMS, fx.BFloat16)
        base_row = block_x * self.block_m + warp_idx * _WMMA_M
        G = self._chunks_per_row
        TPU = self.tiles_per_unit  # ds_store ops per unit == ds_load rounds per unit
        NSL = self.inflight_units  # ring depth (units resident at once)

        # DSCNT is one in-order 6-bit counter for both ds_store and ds_load: op X has
        # retired <=> DSCNT <= issued-1-X. Track each DS op's global index and await a
        # dependency ``dep`` via s_wait_dscnt(clamp(issued-1-dep, 0, 63)).
        issued = 0  # DS ops emitted so far (global issue index)
        unit_last_store = {}  # unit -> gidx of its final ds_store (all TPU stores done)
        unit_loads = {}  # unit -> list of gidx of its TPU ds_loads

        def emit_write(u):
            """Issue unit ``u``'s TPU ds_stores (accumulator -> ring slot u%NSL)."""
            nonlocal issued
            slot = u % NSL
            prev = u - NSL  # last occupant of this slot
            if prev >= 0:
                # WAR: prev unit's re-reads must retire before we overwrite the slot.
                dep = unit_loads[prev][-1]
                rocdl.s_wait_dscnt(max(0, min(_O_DSCNT_MAX, issued - 1 - dep)))
            last = None
            for kk in range(TPU):
                k = u * TPU + kk
                d_col = d_half + fx.Int32(kk * _WMMA_M)  # slot-local column
                bf = o_frags[k].to(fx.BFloat16)
                addr = lds_warp + self._lds_byte(slot, q_st, d_col)
                lds_ptr = buffer_ops.create_llvm_ptr(addr, address_space=3)
                llvm_dialect.store(_ir(bf), lds_ptr, alignment=_CHUNK_BYTES)
                last = issued
                issued += 1
            unit_last_store[u] = last

        def emit_read(u):
            """Re-read unit ``u`` coalesced (ring slot u%NSL) and buffer_store to VRAM."""
            nonlocal issued
            slot = u % NSL
            # RAW: every re-read pulls d-columns spanning all TPU stores of the unit,
            # so wait for the unit's final store before the first load.
            dep = unit_last_store[u]
            rocdl.s_wait_dscnt(max(0, min(_O_DSCNT_MAX, issued - 1 - dep)))
            loaded = []
            gidxs = []
            for r in range(TPU):
                f = fx.Int32(r * _WAVE_LANES) + lane_idx  # 32 b128 chunks per round
                q_out = f // G  # q-row within this warp [0,16)
                d_local = (f % G) * _CHUNK_ELEMS
                addr = lds_warp + self._lds_byte(slot, q_out, d_local)
                lds_ptr = buffer_ops.create_llvm_ptr(addr, address_space=3)
                data = fx.Vector(llvm_dialect.load(v8_ty, lds_ptr))
                loaded.append((data, q_out, d_local))
                gidxs.append(issued)
                issued += 1
            unit_loads[u] = gidxs
            d_base = fx.Int32(u * self.cols_per_tile)  # global d origin of this unit
            for r in range(TPU):
                data, q_out, d_local = loaded[r]
                # RAW: this load's data must be in VGPR before storing it to VRAM.
                rocdl.s_wait_dscnt(max(0, min(_O_DSCNT_MAX, issued - 1 - gidxs[r])))
                pr = base_row + q_out  # global packed row
                q_head = kv_head * self.gqa_ratio + pr % self.gqa_ratio
                seq = pr // self.gqa_ratio
                valid = seq < q_len
                token = q_start + seq
                off_elems = (
                    o_base_elems
                    + token * stride_o_seq
                    + q_head * stride_o_head
                    + d_base
                    + d_local
                )
                off_bytes = off_elems * fx.Int32(_BF16_BYTES)
                off_masked = valid.select(off_bytes, fx.Int32(0x7FFFFFFF))
                buffer_ops.buffer_store(
                    data, o_rsrc, off_masked, mask=None, offset_is_bytes=True
                )

        # Two-stage pipeline: prime unit 0, then overlap unit u+1's write with unit
        # u's read. n_units == 1 collapses to a single write+read with one RAW wait.
        emit_write(0)
        for u in range(self.n_units):
            if u + 1 < self.n_units:
                emit_write(u + 1)
            emit_read(u)
