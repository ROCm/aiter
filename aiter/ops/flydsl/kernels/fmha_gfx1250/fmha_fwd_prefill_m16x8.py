# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MHA Forward Prefill kernel — ``m16x8`` design, gfx1250 (MI400 / mi450).

A fresh, clean FlyDSL kernel written in the high-level layout-algebra style
(tiled copy / tiled MMA + ``SharedAllocator``) — deliberately independent of the
hand-tuned, assembly-mirroring ``fmha_kernel.py`` (inline ASM, raw TDM
descriptors, ``set_vgpr_bank`` hints, per-WMMA schedule tables).

``m16x8`` names the threadgroup shape: **8 waves per threadgroup**. gfx1250 runs
wave32, so a threadgroup is ``8 * 32 = 256`` threads. (The leading ``16`` is the
WMMA M dimension, one 16-row Q sub-tile per wave.)

Layout support — two device kernels over one shared compute core (option B):
  - ``kn_fmha_fwd_prefill_m16x8_thd``  — varlen THD, driven by ``cu_seqlens``.
  - ``kn_fmha_fwd_prefill_m16x8_bshd`` — batched BSHD, uniform ``seq_len`` scalar
    (no ``cu_seqlens`` tensors → nothing transient to bake into a CUDA graph).
Both resolve their per-workgroup base offsets + sequence bounds, then call the
layout-agnostic ``_core_attention`` helper.

Scope — v1 (this file is intentionally config-agnostic in its name):
  - ``qk_hdim == v_hdim == 128``
  - dtype: bf16 for Q/K/V/O
  - grouped-query attention (GQA): ``gqa = nheads_q // nheads_k``
  - causal and non-causal

``qk_hdim``, ``v_hdim`` and the dtype are compile-time (build-time) parameters
captured by the builder closure, so they never appear in the file name and can
be generalized later without changing the runtime kernel signatures.

Target: gfx1250, wave32, 8 waves per threadgroup (256 threads).
"""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import rocdl as rocdl_dialect
from flydsl.expr import arith, buffer_ops, gpu, rocdl
from flydsl.expr.typing import T
from ..tensor_shim import _run_compiled

# ============================================================================
# Threadgroup / arch constants
# ============================================================================

WAVE_SIZE = 32  # gfx1250 kernels run wave32
NUM_WAVES = 8  # "m16x8" — 8 waves per threadgroup
BLOCK_SIZE = WAVE_SIZE * NUM_WAVES  # 256 threads

# "m16x8": each wave owns a 16-row (WMMA M) Q sub-tile → BLOCK_M = 16 * 8 = 128
# Q rows per threadgroup.
WMMA_M = 16
BLOCK_M = WMMA_M * NUM_WAVES  # 128

# v_wmma_f32_16x16x32_bf16/f16: K dimension of one WMMA step.
WMMA_K = 32
BF16_BYTES = 2
Q_CHUNK_ELEMS = 8  # b128 = 8 bf16
Q_CHUNK_BYTES = Q_CHUNK_ELEMS * BF16_BYTES  # 16

# v1 defaults (compile-time; see module docstring).
DEFAULT_QK_HDIM = 128
DEFAULT_V_HDIM = 128
DEFAULT_DTYPE = "bf16"


# ============================================================================
# Small device helpers
# ============================================================================


def _warp_id():
    """Wave (warp) index within the workgroup, matching opus ``waveid_in_workgroup()``."""
    return fx.Int32(rocdl.wave_id())


def _lane_id():
    """Lane index within the wave (wave32), matching opus ``lane_id()``."""
    return fx.Int32(
        rocdl_dialect.mbcnt_lo(T.i32, fx.Int32(-1).ir_value(), fx.Int32(0).ir_value())
    )


def _load_seqlen_pair(ptr_tensor, idx):
    """Load ``ptr_tensor[idx]`` and ``ptr_tensor[idx + 1]`` (adjacent i32s) as one
    ``vector<2xi32>``; returns ``(start, end)`` as ``fx.Int32``.

    The two values are contiguous and the address is uniform (derived from
    ``block_id``), so a single 64-bit load should lower to one ``s_load_b64``.
    """
    p = fx.get_iter(ptr_tensor)
    pair = fx.ptr_load(p + fx.Int64(idx), result_type=fx.Vector.make_type(2, fx.Int32))
    return fx.Int32(pair[0]), fx.Int32(pair[1])


def _packed_tile_indices(gqa_ratio, warp_idx, lane_idx):
    """Map this lane's row in the packed ``(seq, q_head_in_group)`` tile to global
    indices; returns ``(kv_head, q_head_idx, seq_idx)`` (all ``fx.Int32``).

    GQA head x seq packing:
      block_id x -> tile over one kv-head's ``(seq, q_head_in_group)`` plane
      block_id y -> kv_head
    ``q_head_in_group`` is the fast axis, so the ``% / //`` use the small (often
    power-of-two) ``gqa_ratio``. Each of the ``BLOCK_M`` rows is an independent
    query sharing this kv-head's K/V.
    """
    kv_head = fx.Int32(gpu.block_id("y"))
    row_idx = (
        warp_idx * WMMA_M + lane_idx % WMMA_M + fx.Int32(gpu.block_id("x")) * BLOCK_M
    )
    q_head_idx = kv_head * gqa_ratio + row_idx % gqa_ratio
    seq_idx = row_idx // gqa_ratio
    return kv_head, q_head_idx, seq_idx


# ============================================================================
# Q loader (global -> LDS async -> VGPR WMMA fragments)
# ============================================================================


class QManager:
    """Owns everything about Q: its LDS footprint and the global->LDS->VGPR load.

    Keeping this behind one object means the compute core just asks the manager
    how much LDS to reserve (``get_lds_size_in_byte``) and then hands the raw
    allocation base to ``load_q_to_vgpr`` — the per-warp sub-offset and the
    swizzled staging layout are entirely the manager's business. A future Q
    strategy (different tiling / dtype) is a drop-in replacement.

    Staging layout (per warp, tile-major): each of ``qk_hdim // WMMA_K`` K-tiles
    is 16 rows x WMMA_K cols bf16 (1024 B). Within a tile, 4x4 subtiles of
    4 rows x 8 cols; the 8-col b128 chunk index is XOR-swizzled with the 4-row
    subtile index to spread LDS banks. The 8 warps stack their tiles contiguously.
    """

    def __init__(self, *, qk_hdim, gqa_ratio, lds_tiles=None):
        if qk_hdim % WMMA_K != 0:
            raise ValueError(f"qk_hdim must be a multiple of {WMMA_K}; got {qk_hdim}")
        self.qk_hdim = qk_hdim  # compile-time
        self.gqa_ratio = gqa_ratio  # compile-time
        self.k_tiles = qk_hdim // WMMA_K

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
        self._warp_stride = WMMA_M * self.lds_tiles * WMMA_K * BF16_BYTES

    def get_lds_size_in_byte(self):
        """LDS bytes the caller must reserve for Q staging (all 8 warps)."""
        return NUM_WAVES * self._warp_stride

    def _lds_byte(self, row, col_chunk, tile):
        """Swizzled LDS byte offset (within a warp region) for a 8-col b128 chunk."""
        sw = col_chunk ^ (row // 4)  # 4x4 subtile XOR swizzle
        return (
            tile * (WMMA_M * WMMA_K * BF16_BYTES)
            + row * (WMMA_K * BF16_BYTES)
            + sw * Q_CHUNK_BYTES
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
        warp_row0 = fx.Int32(gpu.block_id("x")) * BLOCK_M + warp_idx * WMMA_M

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
                    + fx.Int32(tile * WMMA_K * BF16_BYTES)
                    + col_chunk * Q_CHUNK_BYTES
                )
                self._async_load_vram_to_lds(
                    q_base_i64, g_off, lds_q_base + self._lds_byte(row, col_chunk, slot)
                )

        # Prime the ring: tiles 0..lds_tiles-1 map 1:1 onto slots 0..lds_tiles-1.
        for tile in fx.range_constexpr(self.lds_tiles):
            _issue_async(tile, tile)

        v8_ty = fx.Vector.make_type(Q_CHUNK_ELEMS, fx.BFloat16)
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
            row = lane_idx % WMMA_M
            klane = lane_idx // WMMA_M  # 0 or 1
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
# Shared, layout-agnostic compute core
# ============================================================================


def _core_attention(
    *,
    qk_hdim,
    v_hdim,
    is_causal,
    return_lse,
    gqa_ratio,  # compile-time GQA group size = nheads_q // nheads_kv
    ptr_O,
    ptr_Q,
    ptr_K,
    ptr_V,
    ptr_LSE,
    softmax_scale,
    stride_q_seq,
    stride_k_seq,
    stride_v_seq,
    stride_o_seq,
    stride_q_head,
    stride_k_head,
    stride_v_head,
    stride_o_head,
    # Per-batch token ranges (fx.Int32), resolved by the caller:
    q_start,  # first Q token index of this batch in the global tensor
    q_len,  # valid Q tokens in this batch
    kv_start,  # first K/V token index of this batch
    kv_len,  # valid K/V tokens in this batch
):
    """Layout-agnostic m16x8 compute — empty scaffold.

    Shared by the THD and BSHD kernel entries. The caller resolves the per-batch
    token ranges (``q_start``/``q_len`` and ``kv_start``/``kv_len``) — the only
    part that differs between varlen and batched layouts — and passes them here.
    """
    warp_idx = _warp_id()
    lane_idx = _lane_id()
    kv_head, q_head_idx, seq_idx = _packed_tile_indices(gqa_ratio, warp_idx, lane_idx)

    # ---- Q staging in LDS: the caller only reserves the byte count the manager
    # asks for; the swizzled per-warp layout is entirely QManager's business. ----
    q_mgr = QManager(qk_hdim=qk_hdim, gqa_ratio=gqa_ratio)
    q_smem = fx.SharedAllocator().allocate(q_mgr.get_lds_size_in_byte())
    ptr_lds = fx.Int32(fx.ptrtoint(q_smem.peek().ptr))

    q_frags = q_mgr.load_q_to_vgpr(
        ptr_Q=ptr_Q,
        stride_q_seq=stride_q_seq,
        stride_q_head=stride_q_head,
        q_start=q_start,
        q_len=q_len,
        kv_head=kv_head,
        warp_idx=warp_idx,
        lane_idx=lane_idx,
        ptr_lds=ptr_lds,
        scale=softmax_scale,
    )

    # TODO(fmha_fwd_prefill_m16x8): KV loop (GEMM1 QK using q_frags -> online
    # softmax -> GEMM2 PV) -> epilogue, using (q_head_idx, seq_idx) for the O
    # write and (kv_head, kv_start/kv_len) for K/V. Rows with seq_idx >= q_len
    # (padding tail / varlen) must be masked.
    del q_frags, q_head_idx, seq_idx  # unused until the KV loop / epilogue lands


# ============================================================================
# Builder — one device kernel per (layout, config)
# ============================================================================


@functools.lru_cache(maxsize=None)
def build_fmha_fwd_prefill_m16x8(
    *,
    layout: str = "thd",
    qk_hdim: int = DEFAULT_QK_HDIM,
    v_hdim: int = DEFAULT_V_HDIM,
    dtype_str: str = DEFAULT_DTYPE,
    is_causal: bool = False,
    return_lse: bool = False,
    gqa_ratio: int = 1,
):
    """Build the m16x8 device kernel for a given layout + config.

    ``layout`` is ``"thd"`` (varlen) or ``"bshd"`` (batched). Compile-time
    parameters are captured here and baked into the traced kernel. ``gqa_ratio``
    (= ``nheads_q // nheads_kv``) is compile-time so the per-lane ``% / //`` fold
    to shift/and when it is a power of two.
    """
    assert layout in ("thd", "bshd"), f"layout must be thd|bshd, got {layout!r}"
    # v1 supports a single configuration; generalize later.
    assert (
        qk_hdim == 128 and v_hdim == 128
    ), f"v1 supports qk_hdim == v_hdim == 128 only, got {qk_hdim}/{v_hdim}"
    assert dtype_str == "bf16", f"v1 supports bf16 only, got {dtype_str!r}"
    assert gqa_ratio >= 1, f"gqa_ratio must be >= 1, got {gqa_ratio}"

    QK_HDIM = qk_hdim
    V_HDIM = v_hdim
    CAUSAL = bool(is_causal)
    RET_LSE = bool(return_lse)
    GQA_RATIO = int(gqa_ratio)

    if layout == "thd":

        @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
        def kn_fmha_fwd_prefill_m16x8_thd(
            ptr_O: fx.Pointer,
            ptr_Q: fx.Pointer,
            ptr_K: fx.Pointer,
            ptr_V: fx.Pointer,
            ptr_LSE: fx.Pointer,
            ptr_cu_seqlens_q: fx.Pointer,
            ptr_cu_seqlens_k: fx.Pointer,
            softmax_scale: fx.Float32,
            stride_q_seq: fx.Int32,
            stride_k_seq: fx.Int32,
            stride_v_seq: fx.Int32,
            stride_o_seq: fx.Int32,
            stride_q_head: fx.Int32,
            stride_k_head: fx.Int32,
            stride_v_head: fx.Int32,
            stride_o_head: fx.Int32,
            max_seqlen_q: fx.Int32,
            max_seqlen_k: fx.Int32,
        ):
            """Varlen THD entry — empty scaffold.

            THD: this batch's token ranges come from cu_seqlens (batch = grid.z).
            """
            batch = fx.Int32(gpu.block_id("z"))
            q_start, q_end = _load_seqlen_pair(ptr_cu_seqlens_q, batch)
            kv_start, kv_end = _load_seqlen_pair(ptr_cu_seqlens_k, batch)
            q_len = q_end - q_start
            kv_len = kv_end - kv_start

            _core_attention(
                qk_hdim=QK_HDIM,
                v_hdim=V_HDIM,
                is_causal=CAUSAL,
                return_lse=RET_LSE,
                gqa_ratio=GQA_RATIO,
                ptr_O=ptr_O,
                ptr_Q=ptr_Q,
                ptr_K=ptr_K,
                ptr_V=ptr_V,
                ptr_LSE=ptr_LSE,
                softmax_scale=softmax_scale,
                stride_q_seq=stride_q_seq,
                stride_k_seq=stride_k_seq,
                stride_v_seq=stride_v_seq,
                stride_o_seq=stride_o_seq,
                stride_q_head=stride_q_head,
                stride_k_head=stride_k_head,
                stride_v_head=stride_v_head,
                stride_o_head=stride_o_head,
                q_start=q_start,
                q_len=q_len,
                kv_start=kv_start,
                kv_len=kv_len,
            )

        return kn_fmha_fwd_prefill_m16x8_thd

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def kn_fmha_fwd_prefill_m16x8_bshd(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
        softmax_scale: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
    ):
        """Batched BSHD entry — empty scaffold.

        Uniform sequence lengths (``seq_len_q`` / ``seq_len_k``) replace
        cu_seqlens — nothing transient, so this path is CUDA-graph safe.
        Token base is batch_idx * seq_len (batch = grid.z).
        """
        batch = fx.Int32(gpu.block_id("z"))

        _core_attention(
            qk_hdim=QK_HDIM,
            v_hdim=V_HDIM,
            is_causal=CAUSAL,
            return_lse=RET_LSE,
            gqa_ratio=GQA_RATIO,
            ptr_O=ptr_O,
            ptr_Q=ptr_Q,
            ptr_K=ptr_K,
            ptr_V=ptr_V,
            ptr_LSE=ptr_LSE,
            softmax_scale=softmax_scale,
            stride_q_seq=stride_q_seq,
            stride_k_seq=stride_k_seq,
            stride_v_seq=stride_v_seq,
            stride_o_seq=stride_o_seq,
            stride_q_head=stride_q_head,
            stride_k_head=stride_k_head,
            stride_v_head=stride_v_head,
            stride_o_head=stride_o_head,
            q_start=batch * seq_len_q,
            q_len=seq_len_q,
            kv_start=batch * seq_len_k,
            kv_len=seq_len_k,
        )

    return kn_fmha_fwd_prefill_m16x8_bshd


# ============================================================================
# Launch wrappers + host entries
# ============================================================================
# NOTE: v1 scaffold. The device kernel bodies are empty, so launching produces
# no output yet — this wiring exists so the dispatch paths in fmha_kernels.py can
# route qk_hdim==128 here while the kernel is being built out.

_launch_fns = {}  # {(layout, is_causal, return_lse, gqa_ratio): @flyc.jit launch fn}


def _ensure_thd_kernel(is_causal: bool, return_lse: bool, gqa_ratio: int):
    key = ("thd", bool(is_causal), bool(return_lse), int(gqa_ratio))
    if key in _launch_fns:
        return
    kernel = build_fmha_fwd_prefill_m16x8(
        layout="thd", is_causal=is_causal, return_lse=return_lse, gqa_ratio=gqa_ratio
    )

    @flyc.jit
    def _launch(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
        ptr_cu_seqlens_q: fx.Pointer,
        ptr_cu_seqlens_k: fx.Pointer,
        softmax_scale: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        num_heads_kv: fx.Int32,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        # 3D grid: x = tiles over (seq, q_head_in_group) per kv-head,
        #          y = kv_head, z = batch. block = 256 (8 waves x wave32).
        grid_x = arith.index_cast(
            T.index,
            arith.ceildivui(
                arith.unwrap(max_seqlen_q * gqa_ratio),
                arith.constant(BLOCK_M, type=T.i32),
            ),
        )
        grid_y = arith.index_cast(T.index, num_heads_kv)
        grid_z = arith.index_cast(T.index, batch_size)

        launcher = kernel(
            ptr_O,
            ptr_Q,
            ptr_K,
            ptr_V,
            ptr_LSE,
            ptr_cu_seqlens_q,
            ptr_cu_seqlens_k,
            softmax_scale,
            stride_q_seq,
            stride_k_seq,
            stride_v_seq,
            stride_o_seq,
            stride_q_head,
            stride_k_head,
            stride_v_head,
            stride_o_head,
            max_seqlen_q,
            max_seqlen_k,
        )
        launcher.launch(
            grid=(grid_x, grid_y, grid_z),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _launch_fns[key] = _launch


def _ensure_bshd_kernel(is_causal: bool, return_lse: bool, gqa_ratio: int):
    key = ("bshd", bool(is_causal), bool(return_lse), int(gqa_ratio))
    if key in _launch_fns:
        return
    kernel = build_fmha_fwd_prefill_m16x8(
        layout="bshd", is_causal=is_causal, return_lse=return_lse, gqa_ratio=gqa_ratio
    )

    @flyc.jit
    def _launch(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
        softmax_scale: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        num_heads_kv: fx.Int32,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        # 3D grid: x = tiles over (seq, q_head_in_group) per kv-head,
        #          y = kv_head, z = batch. block = 256 (8 waves x wave32).
        grid_x = arith.index_cast(
            T.index,
            arith.ceildivui(
                arith.unwrap(seq_len_q * gqa_ratio),
                arith.constant(BLOCK_M, type=T.i32),
            ),
        )
        grid_y = arith.index_cast(T.index, num_heads_kv)
        grid_z = arith.index_cast(T.index, batch_size)

        launcher = kernel(
            ptr_O,
            ptr_Q,
            ptr_K,
            ptr_V,
            ptr_LSE,
            softmax_scale,
            stride_q_seq,
            stride_k_seq,
            stride_v_seq,
            stride_o_seq,
            stride_q_head,
            stride_k_head,
            stride_v_head,
            stride_o_head,
            seq_len_q,
            seq_len_k,
        )
        launcher.launch(
            grid=(grid_x, grid_y, grid_z),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _launch_fns[key] = _launch


def flash_attn_varlen_m16x8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale=None,
    causal=False,
    out=None,
    return_lse=False,
):
    """Host entry — varlen THD, qk_hdim=v_hdim=128, bf16.

    v1 scaffold: builds and launches the empty THD kernel and returns the
    (unwritten) output tensor.
    """
    assert q.dtype == torch.bfloat16, f"Expected bf16, got {q.dtype}"
    assert q.shape[-1] == 128, f"Expected qk_hdim=128, got {q.shape[-1]}"
    assert v.shape[-1] == 128, f"Expected v_hdim=128, got {v.shape[-1]}"

    total_q_tokens = q.shape[0]
    batch = cu_seqlens_q.shape[0] - 1
    nheads_q = q.shape[1]
    nheads_k = k.shape[1]
    gqa = nheads_q // nheads_k

    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    if out is None:
        out = torch.empty(
            (total_q_tokens, nheads_q, 128), dtype=torch.bfloat16, device=q.device
        )
    if return_lse:
        lse = torch.empty(
            (total_q_tokens, nheads_q), dtype=torch.float32, device=q.device
        )
    else:
        lse = torch.empty(
            (batch, nheads_q, max_seqlen_q), dtype=torch.float32, device=q.device
        )

    # Byte strides for Q/K/V, element strides for O — matches the gfx1250 fmha
    # family convention; revisit when the clean kernel body is implemented.
    bpp = q.element_size()
    stride_q_seq = q.stride(0) * bpp
    stride_k_seq = k.stride(0) * bpp
    stride_v_seq = v.stride(0) * bpp
    stride_o_seq = out.stride(0)
    stride_q_head = q.stride(1) * bpp
    stride_k_head = k.stride(1) * bpp
    stride_v_head = v.stride(1) * bpp
    stride_o_head = out.stride(1)

    _ensure_thd_kernel(bool(causal), bool(return_lse), gqa)

    _run_compiled(
        _launch_fns[("thd", bool(causal), bool(return_lse), gqa)],
        out,
        q,
        k,
        v,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale,
        stride_q_seq,
        stride_k_seq,
        stride_v_seq,
        stride_o_seq,
        stride_q_head,
        stride_k_head,
        stride_v_head,
        stride_o_head,
        max_seqlen_q,
        max_seqlen_k,
        nheads_k,
        batch,
        torch.cuda.current_stream(),
    )

    if return_lse:
        return out, lse
    return out


def flash_attn_batch_m16x8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale=None,
    causal=False,
    out=None,
    return_lse=False,
):
    """Host entry — batched BSHD ``[B, S, H, D]``, qk_hdim=v_hdim=128, bf16.

    Uses the dedicated BSHD kernel with a uniform ``seq_len`` scalar (no
    cu_seqlens), so there is nothing transient to bake into a CUDA graph.

    v1 scaffold: builds and launches the empty BSHD kernel and returns the
    (unwritten) output tensor.
    """
    assert q.dtype == torch.bfloat16, f"Expected bf16, got {q.dtype}"
    assert q.dim() == 4, f"Expected 4D BSHD tensor, got rank {q.dim()}"
    assert q.shape[-1] == 128, f"Expected qk_hdim=128, got {q.shape[-1]}"
    assert v.shape[-1] == 128, f"Expected v_hdim=128, got {v.shape[-1]}"

    batch, seq_len_q, nheads_q, _ = q.shape
    seq_len_k = k.shape[1]
    nheads_k = k.shape[2]
    gqa = nheads_q // nheads_k

    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    if out is None:
        out = torch.empty(
            (batch, seq_len_q, nheads_q, 128), dtype=torch.bfloat16, device=q.device
        )
    # LSE is [B, H, S_q] (scratch when return_lse is False).
    lse = torch.empty(
        (batch, nheads_q, seq_len_q), dtype=torch.float32, device=q.device
    )

    # Byte strides for Q/K/V, element strides for O. BSHD: seq is dim 1, head
    # dim 2 — the per-batch base is derived in-kernel as batch_idx * seq_len.
    bpp = q.element_size()
    stride_q_seq = q.stride(1) * bpp
    stride_k_seq = k.stride(1) * bpp
    stride_v_seq = v.stride(1) * bpp
    stride_o_seq = out.stride(1)
    stride_q_head = q.stride(2) * bpp
    stride_k_head = k.stride(2) * bpp
    stride_v_head = v.stride(2) * bpp
    stride_o_head = out.stride(2)

    _ensure_bshd_kernel(bool(causal), bool(return_lse), gqa)

    _run_compiled(
        _launch_fns[("bshd", bool(causal), bool(return_lse), gqa)],
        out,
        q,
        k,
        v,
        lse,
        softmax_scale,
        stride_q_seq,
        stride_k_seq,
        stride_v_seq,
        stride_o_seq,
        stride_q_head,
        stride_k_head,
        stride_v_head,
        stride_o_head,
        seq_len_q,
        seq_len_k,
        nheads_k,
        batch,
        torch.cuda.current_stream(),
    )

    if return_lse:
        return out, lse
    return out
