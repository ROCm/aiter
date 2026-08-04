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

# Q/K/V staging managers (own their LDS swizzles + async copy schedules). They are
# self-contained: this kernel maintains its own arch constants below and passes the
# config each manager needs through its constructor.
from .mha_buffer_managers import QManager16b, KManager16b, VManager16b, OManager16b

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

# v1 defaults (compile-time; see module docstring).
DEFAULT_QK_HDIM = 128
DEFAULT_V_HDIM = 128
DEFAULT_DTYPE = "bf16"

# KV sequence block (columns of one QK GEMM tile). Configurable; 64 for now.
N_BLOCK_CHOICES = (32, 64, 128, 256)
DEFAULT_N_BLOCK = 64

# Ping-pong K LDS buffers the main loop rotates through (double-buffered prefetch).
N_KV_PP = 2

# NOTE: the WMMA/tiling constants (WMMA_K, chunk sizes, K/V write-tile + V swizzle
# granularity) live inside mha_buffer_managers.py — they are intrinsic to the
# managers' LDS layouts, so the kernel no longer declares them here.


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
# Shared, layout-agnostic compute core
# ============================================================================


def _core_attention(
    *,
    qk_hdim,
    v_hdim,
    n_block,  # compile-time KV block width (columns of one QK GEMM tile)
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

    # One SharedAllocator per kernel (flydsl constraint); Q/K/V carve their regions
    # from it. Each manager only reports the byte count it needs — the swizzled
    # layout inside each region is the manager's own business.
    smem = fx.SharedAllocator()

    # ---- Q staging in LDS. ----
    q_mgr = QManager16b(qk_hdim=qk_hdim, gqa_ratio=gqa_ratio, num_waves=NUM_WAVES)
    q_smem = smem.allocate(q_mgr.get_lds_size_in_byte())
    q_lds_base = fx.Int32(fx.ptrtoint(q_smem.peek().ptr))

    q_frags = q_mgr.load_q_to_vgpr(
        ptr_Q=ptr_Q,
        stride_q_seq=stride_q_seq,
        stride_q_head=stride_q_head,
        q_start=q_start,
        q_len=q_len,
        kv_head=kv_head,
        block_x=fx.Int32(gpu.block_id("x")),
        warp_idx=warp_idx,
        lane_idx=lane_idx,
        ptr_lds=q_lds_base,
        scale=softmax_scale,
    )

    # ---- K staging: N_KV_PP ping-pong buffers the main loop rotates through.
    # KManager owns only one block's swizzle/size; the ring is the caller's. ----
    k_mgr = KManager16b(qk_hdim=qk_hdim, n_block=n_block, num_waves=NUM_WAVES)
    k_blk_bytes = k_mgr.get_lds_size_in_byte()
    k_smem = smem.allocate(N_KV_PP * k_blk_bytes)
    k_lds_base = fx.Int32(fx.ptrtoint(k_smem.peek().ptr))

    def _k_lds_buf(pp):  # base of ping-pong buffer ``pp`` in [0, N_KV_PP)
        return k_lds_base + fx.Int32(pp * k_blk_bytes)

    def _kv_block_valid(blk_row0):
        # Rows [blk_row0, blk_row0+n_block) of the batch's kv range; how many are
        # in-bounds relative to the block start (0..n_block). TODO(causal): fold
        # the per-M-block causal upper bound into kv_len before this.
        rem = kv_len - blk_row0
        return (rem < fx.Int32(n_block)).select(rem, fx.Int32(n_block))

    # ---- V staging: own swizzle (transpose-load friendly), N_KV_PP ping-pong. ----
    v_mgr = VManager16b(v_hdim=v_hdim, n_block=n_block, num_waves=NUM_WAVES)
    v_blk_bytes = v_mgr.get_lds_size_in_byte()
    v_smem = smem.allocate(N_KV_PP * v_blk_bytes)
    v_lds_base = fx.Int32(fx.ptrtoint(v_smem.peek().ptr))

    def _v_lds_buf(pp):  # base of ping-pong buffer ``pp`` in [0, N_KV_PP)
        return v_lds_base + fx.Int32(pp * v_blk_bytes)

    # Prologue: prime buffer 0 with the first K and V blocks so the main loop can
    # enter assuming its current block is already resident (async in flight).
    k_mgr.async_load_vram_to_lds(
        ptr_lds=_k_lds_buf(0),
        ptr_K=ptr_K,
        stride_k_seq=stride_k_seq,
        stride_k_head=stride_k_head,
        kv_head=kv_head,
        kv_row0=kv_start,
        kv_valid=_kv_block_valid(fx.Int32(0)),
        warp_idx=warp_idx,
        lane_idx=lane_idx,
    )
    v_mgr.async_load_vram_to_lds(
        ptr_lds=_v_lds_buf(0),
        ptr_V=ptr_V,
        stride_v_seq=stride_v_seq,
        stride_v_head=stride_v_head,
        kv_head=kv_head,
        kv_row0=kv_start,
        kv_valid=_kv_block_valid(fx.Int32(0)),
        warp_idx=warp_idx,
        lane_idx=lane_idx,
    )

    # TODO(fmha_fwd_prefill_m16x8): main KV loop over blocks — wait on the current
    # buffer's async, prefetch the next block into the next ping-pong buffer, then
    # GEMM1 QK (q_frags x K) -> online softmax -> GEMM2 PV -> epilogue. Uses
    # (q_head_idx, seq_idx) for the O write; rows with seq_idx >= q_len are masked.
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
    n_block: int = DEFAULT_N_BLOCK,
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
    assert n_block in N_BLOCK_CHOICES, f"n_block must be in {N_BLOCK_CHOICES}, got {n_block}"

    QK_HDIM = qk_hdim
    V_HDIM = v_hdim
    N_BLOCK = int(n_block)
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
                n_block=N_BLOCK,
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
            n_block=N_BLOCK,
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
