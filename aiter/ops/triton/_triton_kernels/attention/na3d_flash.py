# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Triton flash-style 3D neighborhood attention.

Each query token attends to the (KT x KH x KW) neighborhood centered on its
(t, h, w) grid position, with inward border-shift so every window has the full
KT x KH x KW keys.

Algorithm:
- Grid dim[0] = T × H × ⌈W / BLOCK_Q⌉; each program covers one (t, h) row.
- Q is loaded once into registers and held for all KT x KH inner iterations.
- Each iteration loads one (t_kv, h_kv) row of K and V at the shared W window.
- Running online softmax in log2 space (exp2, log2(e) folded into scale).
- Tile (BLOCK_Q, BLOCK_KV, num_warps, num_stages) is read from the per-arch config
  configs/<arch>/triton/attention/na3d_flash/DEFAULT.json (no runtime autotune).
"""

import functools

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir


@functools.cache
def _get_config(KW: int) -> dict:
    """Per-arch tile from configs/<arch>/triton/attention/na3d_flash/DEFAULT.json.

    Two published entries cover the two KW families:
      - na3d_flash_small_kw (BLOCK_Q=16, BLOCK_KV=32) for KW <= 17
      - na3d_flash_large_kw (BLOCK_Q=32, BLOCK_KV=64) for KW in [18, 33]
    BLOCK_KV must satisfy BLOCK_KV >= BLOCK_Q + KW - 1 so the KV tile covers the
    union of all BLOCK_Q query windows; both entries meet that at their KW bound.
    """
    cfg_dir = resolve_config_dir("attention", "NA3D_FLASH", backend="triton")
    config = load_config_json(f"{cfg_dir}/DEFAULT.json")
    return config["na3d_flash_small_kw" if KW <= 17 else "na3d_flash_large_kw"]


@functools.cache
def _get_config_3d(KT: int, KH: int, KW: int, HD: int) -> dict:
    """Per-arch 3D-tiled config from configs/<arch>/triton/attention/na3d_flash/DEFAULT.json.

    Selects wave32 3D query tiles ``(BT, BH, BW)`` matched to window aspect ratio
    and head dimension so the KV union volume and register footprint stay minimal:
      - na3d_flash_3d_large_hd : HD > 64 (num_stages=1 avoids VGPR spills)
      - na3d_flash_3d_large_kt : KT > 5 (balanced 4x2x2 cube for deep windows)
      - na3d_flash_3d_small_k  : KT <= 3, KH <= 5, KW <= 5 (2x4x4 for higher KV reuse)
      - na3d_flash_3d_medium_k : remaining KT <= 5 windows (2x2x4, BKV=16, nw=1)
    """
    cfg_dir = resolve_config_dir("attention", "NA3D_FLASH", backend="triton")
    config = load_config_json(f"{cfg_dir}/DEFAULT.json")
    if HD > 64:
        key = "na3d_flash_3d_large_hd"
    elif KT > 5:
        key = "na3d_flash_3d_large_kt"
    elif KT <= 3 and KH <= 5 and KW <= 5:
        key = "na3d_flash_3d_small_k"
    else:
        key = "na3d_flash_3d_medium_k"
    return config[key]


_na3d_flash_fwd_repr = make_kernel_repr(
    "_na3d_flash_fwd",
    ["BLOCK_Q", "BLOCK_KV", "KT", "KH", "KW", "HD"],
)

_na3d_flash_fwd_3d_repr = make_kernel_repr(
    "_na3d_flash_fwd_3d",
    ["BT", "BH", "BW", "BLOCK_KV", "KT", "KH", "KW", "HD", "SWIZZLE_T", "SWIZZLE_H"],
)


@triton.jit(repr=_na3d_flash_fwd_repr)
def _na3d_flash_fwd(
    Q_ptr,
    K_ptr,
    V_ptr,
    Out_ptr,
    stride_b: tl.int64,  # B stride   = SEQ * NH * HD  (elements between batches)
    stride_nh: tl.int64,  # NH stride  = HD              (elements between heads)
    stride_seq: tl.int64,  # seq stride = NH * HD         (elements between tokens)
    NH,  # number of attention heads (runtime)
    T,
    H,
    W,
    HD: tl.constexpr,  # head dimension: any power-of-2 value
    KT: tl.constexpr,  # neighborhood depth: constexpr -> static loop bound
    KH: tl.constexpr,  # neighborhood height: constexpr -> static loop bound
    KW: tl.constexpr,  # neighborhood width: used in W-direction masking
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,  # power-of-2, >= BLOCK_Q + KW - 1
):
    """Flash-attention inner kernel. Grid dim[0] = T*H*ceil(W/BLOCK_Q).

    Operates on (B, T, H, W, NH, HD) layout.
    Each program is assigned to exactly one (t, h) row via
    decomposition of pid_q into (row_idx, w_block_idx).

    Addressing: for batch b and head nh,
        element (b, t, h, w, nh, hd) = ptr + b*stride_b + seq*stride_seq + nh*stride_nh + hd
    where seq = t*H*W + h*W + w.
    """
    pid_q = tl.program_id(0)
    pid_bnh = tl.program_id(1)
    W64 = W.to(tl.int64)
    HW = H * W64  # promotes to int64

    # Decompose pid_q into (t,h) row and W-block within that row.
    W_blocks = (W + BLOCK_Q - 1) // BLOCK_Q  # programs per (t,h) row
    row_idx = pid_q // W_blocks  # which (t,h) row  (t*H + h)
    w_bid = pid_q % W_blocks  # which W-block in that row

    prog_t = row_idx // H  # scalar: same t for every query in this program
    prog_h = row_idx % H  # scalar: same h for every query in this program

    q_offs = tl.arange(0, BLOCK_Q)
    q_w = w_bid * BLOCK_Q + q_offs  # W positions for this block
    q_mask = q_w < W  # last block in a row may be partial

    q_idx = (
        prog_t.to(tl.int64) * HW + prog_h.to(tl.int64) * W64 + q_w.to(tl.int64)
    )  # for Q load and Out store
    # Inward-shifted centered neighborhood window starts.
    # prog_t / prog_h are scalars; t_ws / h_ws need no tl.min reduction.
    q_w_ws = tl.minimum(tl.maximum(q_w - KW // 2, 0), W - KW)

    t_ws = tl.minimum(tl.maximum(prog_t - KT // 2, 0), T - KT)  # scalar
    h_ws = tl.minimum(tl.maximum(prog_h - KH // 2, 0), H - KH)  # scalar
    w_lo = tl.min(tl.where(q_mask, q_w_ws, W))

    hd_offs = tl.arange(0, HD)
    kv_offs = tl.arange(0, BLOCK_KV)

    # Decompose pid_bnh into (b, nh) for base address.
    # int64: stride_b = SEQ * NH * HD can exceed int32 range for large volumes.
    b_idx = pid_bnh // NH
    nh_idx = pid_bnh % NH
    base = b_idx.to(tl.int64) * stride_b.to(tl.int64) + nh_idx.to(
        tl.int64
    ) * stride_nh.to(tl.int64)
    kv_w = w_lo + kv_offs
    kv_ok = kv_w < W  # W-boundary guard, constant across the KT x KH loop

    # Q loaded once into registers for all KT x KH iterations.
    Q_tile = tl.load(
        Q_ptr + base + q_idx[:, None] * stride_seq + hd_offs[None, :],
        mask=q_mask[:, None],
        other=0.0,
    ).to(tl.bfloat16)

    m_i = tl.full((BLOCK_Q,), -3e38, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, HD), dtype=tl.float32)

    # Flat KT x KH loop with static trip count - enables Triton's software pipeliner.
    # valid_w, kv_ok, and q_mask are loop-invariant; the compiler's LICM handles them
    # without explicit pre-computation (pre-computing would raise VGPR pressure and
    # reduce occupancy on CDNA).
    for kv_idx in range(KT * KH):
        dt = kv_idx // KH
        dh = kv_idx % KH
        t_kv = t_ws + dt
        h_kv = h_ws + dh

        row_base = t_kv * HW + h_kv.to(tl.int64) * W64
        kv_flat = row_base + kv_w.to(tl.int64)

        K_T = tl.load(
            K_ptr + base + kv_flat[None, :] * stride_seq + hd_offs[:, None],
            mask=kv_ok[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        V_tile = tl.load(
            V_ptr + base + kv_flat[:, None] * stride_seq + hd_offs[None, :],
            mask=kv_ok[:, None],
            other=0.0,
        ).to(tl.bfloat16)

        # QK in log2 space: fold log2(e) into the scale to use exp2 throughout.
        scores = tl.dot(Q_tile, K_T, out_dtype=tl.float32) * 1.4426950408889634

        valid_w = (kv_w[None, :] >= q_w_ws[:, None]) & (
            kv_w[None, :] < q_w_ws[:, None] + KW
        )
        scores = tl.where(
            valid_w & kv_ok[None, :] & q_mask[:, None], scores, float("-inf")
        )

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        exp_s = tl.exp2(scores - m_new[:, None])
        exp_diff = tl.exp2(m_i - m_new)
        l_i = l_i * exp_diff + tl.sum(exp_s, axis=1)
        acc = acc * exp_diff[:, None] + tl.dot(
            exp_s.to(tl.bfloat16), V_tile, out_dtype=tl.float32
        )
        m_i = m_new

    safe_l = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / safe_l[:, None]

    tl.store(
        Out_ptr + base + q_idx[:, None] * stride_seq + hd_offs[None, :],
        acc.to(tl.bfloat16),
        mask=q_mask[:, None],
    )


@triton.jit(repr=_na3d_flash_fwd_3d_repr)
def _na3d_flash_fwd_3d(
    Q_ptr,
    K_ptr,
    V_ptr,
    Out_ptr,
    q_stride_b: tl.int64,
    q_stride_nh: tl.int64,
    q_stride_seq: tl.int64,
    k_stride_b: tl.int64,
    k_stride_nh: tl.int64,
    k_stride_seq: tl.int64,
    v_stride_b: tl.int64,
    v_stride_nh: tl.int64,
    v_stride_seq: tl.int64,
    o_stride_b: tl.int64,
    o_stride_nh: tl.int64,
    o_stride_seq: tl.int64,
    NH,
    T,
    H,
    W,
    HD: tl.constexpr,
    KT: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    SWIZZLE_T: tl.constexpr,
    SWIZZLE_H: tl.constexpr,
    USE_I32_REL: tl.constexpr,
    NO_DIVMOD_2D: tl.constexpr,
):
    """3D-tiled wave32-optimized flash 3D neighborhood attention (gfx1151).

    Key differences from the 1D CDNA kernel:
    - Grid is ``(B * NH, cdiv(T, BT) * cdiv(H, BH) * cdiv(W, BW))`` with 2D/3D
      ``(SWIZZLE_T, SWIZZLE_H)`` swizzling so heads and neighboring temporal/spatial
      blocks share channels-last K/V tiles in L2/MALL.
    - Each program processes a 3D query patch ``(BT, BH, BW)`` and iterates over
      the tight 3D KV union ``(BT+KT-1, BH+KH-1, BW+KW-1)`` with ``dt`` hoisted
      to the outer loop, reducing masked WMMA flops and KV reloads.
    - ``V_tile`` loads are issued after ``Q @ K^T`` so ``K_tile`` VGPRs are freed
      immediately while ``V_tile`` latency overlaps online softmax.
    - Independent per-tensor strides allow zero-copy consumption of strided
      QKV slices (e.g., ``qkv.chunk(3, dim=-1)[2].view(...)``).
    """
    pid_bnh = tl.program_id(0)
    pid_s = tl.program_id(1)

    W64 = W.to(tl.int64)
    HW64 = H * W64
    T_blocks = (T + BT - 1) // BT
    H_blocks = (H + BH - 1) // BH
    W_blocks = (W + BW - 1) // BW

    if SWIZZLE_T > 1:
        # 3D cube swizzle: group SWIZZLE_T temporal slabs x SWIZZLE_H row blocks
        # along W so temporally overlapping frames stay hot in MALL/L2.
        gt_id = pid_s // (SWIZZLE_T * H_blocks * W_blocks)
        rem_t = pid_s % (SWIZZLE_T * H_blocks * W_blocks)
        t_gs = tl.minimum(SWIZZLE_T, T_blocks - gt_id * SWIZZLE_T)
        gh_id = rem_t // (t_gs * SWIZZLE_H * W_blocks)
        rem_h = rem_t % (t_gs * SWIZZLE_H * W_blocks)
        h_gs = tl.minimum(SWIZZLE_H, H_blocks - gh_id * SWIZZLE_H)
        w_bid = rem_h // (t_gs * h_gs)
        rem_th = rem_h % (t_gs * h_gs)
        t_bid = gt_id * SWIZZLE_T + (rem_th // h_gs)
        h_bid = gh_id * SWIZZLE_H + (rem_th % h_gs)
    elif SWIZZLE_H > 1:
        HW_blocks = H_blocks * W_blocks
        t_bid = pid_s // HW_blocks
        rem = pid_s % HW_blocks
        gh_id = rem // (SWIZZLE_H * W_blocks)
        rem2 = rem % (SWIZZLE_H * W_blocks)
        h_group_size = tl.minimum(SWIZZLE_H, H_blocks - gh_id * SWIZZLE_H)
        w_bid = rem2 // h_group_size
        h_bid = gh_id * SWIZZLE_H + (rem2 % h_group_size)
    else:
        w_bid = pid_s % W_blocks
        rem = pid_s // W_blocks
        h_bid = rem % H_blocks
        t_bid = rem // H_blocks

    BQ: tl.constexpr = BT * BH * BW
    q_lin = tl.arange(0, BQ)
    q_dw = q_lin % BW
    q_dh = (q_lin // BW) % BH
    q_dt = q_lin // (BH * BW)

    q_t = t_bid * BT + q_dt
    q_h = h_bid * BH + q_dh
    q_w = w_bid * BW + q_dw
    q_mask = (q_t < T) & (q_h < H) & (q_w < W)

    # Scalar union box origin; clamping at borders never expands the union box.
    t_lo = tl.minimum(tl.maximum(t_bid * BT - KT // 2, 0), T - KT)
    h_lo = tl.minimum(tl.maximum(h_bid * BH - KH // 2, 0), H - KH)
    w_lo = tl.minimum(tl.maximum(w_bid * BW - KW // 2, 0), W - KW)

    # Relative window starts within the union box; out-of-bounds queries receive
    # a sentinel so unsigned range comparisons naturally reject them.
    q_t_rws = tl.minimum(tl.maximum(q_t - KT // 2, 0), T - KT) - t_lo
    q_h_rws = tl.minimum(tl.maximum(q_h - KH // 2, 0), H - KH) - h_lo
    q_w_rws = tl.where(
        q_mask,
        tl.minimum(tl.maximum(q_w - KW // 2, 0), W - KW) - w_lo,
        1048576,
    )

    t_lim = T - t_lo
    h_lim = H - h_lo
    w_lim = W - w_lo

    b_idx = pid_bnh // NH
    nh_idx = pid_bnh % NH
    q_base = b_idx * q_stride_b + nh_idx * q_stride_nh
    o_base = b_idx * o_stride_b + nh_idx * o_stride_nh
    lo_flat = t_lo * HW64 + h_lo * W64 + w_lo
    K_base_ptr = K_ptr + (
        b_idx * k_stride_b + nh_idx * k_stride_nh + lo_flat * k_stride_seq
    )
    V_base_ptr = V_ptr + (
        b_idx * v_stride_b + nh_idx * v_stride_nh + lo_flat * v_stride_seq
    )

    q_idx = q_t * HW64 + q_h * W64 + q_w
    hd_offs = tl.arange(0, HD)

    Q_tile = tl.load(
        Q_ptr + q_base + q_idx[:, None] * q_stride_seq + hd_offs[None, :],
        mask=q_mask[:, None],
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.bfloat16)

    m_i = tl.full((BQ,), -3e38, dtype=tl.float32)
    l_i = tl.zeros((BQ,), dtype=tl.float32)
    acc = tl.zeros((BQ, HD), dtype=tl.float32)

    TU: tl.constexpr = BT + KT - 1
    HU: tl.constexpr = BH + KH - 1
    WU: tl.constexpr = BW + KW - 1
    HW_U: tl.constexpr = HU * WU
    N_HW_ITERS: tl.constexpr = (HW_U + BLOCK_KV - 1) // BLOCK_KV
    STEP_DH: tl.constexpr = BLOCK_KV // WU
    STEP_DW: tl.constexpr = BLOCK_KV % WU
    kv_offs = tl.arange(0, BLOCK_KV)

    if USE_I32_REL:
        k_ss_r = k_stride_seq.to(tl.int32)
        v_ss_r = v_stride_seq.to(tl.int32)
        W_r = W.to(tl.int32)
    else:
        k_ss_r = k_stride_seq
        v_ss_r = v_stride_seq
        W_r = W64

    h_bound = tl.minimum(h_lim, HU)

    for dt in range(TU):
        if dt < t_lim:
            K_t_ptr = K_base_ptr + dt * HW64 * k_stride_seq
            V_t_ptr = V_base_ptr + dt * HW64 * v_stride_seq
            if BT > 1:
                valid_t = ((dt - q_t_rws).to(tl.uint32) < KT) & q_mask
            else:
                valid_t = q_mask
            q_w_rws_t = tl.where(valid_t, q_w_rws, 1048576)

            if NO_DIVMOD_2D:
                kv_dw = kv_offs % WU
                kv_dh = kv_offs // WU

            for it in range(N_HW_ITERS):
                if not NO_DIVMOD_2D:
                    kv_lin = it * BLOCK_KV + kv_offs
                    kv_dw = kv_lin % WU
                    kv_dh = kv_lin // WU

                kv_ok = (kv_dh < h_bound) & (kv_dw < w_lim)
                kv_off = kv_dh * W_r + kv_dw

                K_tile = tl.load(
                    K_t_ptr + kv_off[:, None] * k_ss_r + hd_offs[None, :],
                    mask=kv_ok[:, None],
                    other=0.0,
                ).to(tl.bfloat16)

                scores = (
                    tl.dot(Q_tile, tl.trans(K_tile), out_dtype=tl.float32)
                    * 1.4426950408889634
                )

                # Load V after Q@K^T so K_tile VGPRs can be reused while V load
                # latency overlaps the masking and online-softmax instructions.
                V_tile = tl.load(
                    V_t_ptr + kv_off[:, None] * v_ss_r + hd_offs[None, :],
                    mask=kv_ok[:, None],
                    other=0.0,
                ).to(tl.bfloat16)

                kv_dw_m = tl.where(kv_ok, kv_dw, 2097152)
                valid = (kv_dw_m[None, :] - q_w_rws_t[:, None]).to(tl.uint32) < KW
                if BH > 1:
                    valid = valid & (
                        (kv_dh[None, :] - q_h_rws[:, None]).to(tl.uint32) < KH
                    )
                scores = tl.where(valid, scores, float("-inf"))

                m_new = tl.maximum(m_i, tl.max(scores, axis=1))
                exp_s = tl.exp2(scores - m_new[:, None])
                exp_diff = tl.exp2(m_i - m_new)
                l_i = l_i * exp_diff + tl.sum(exp_s, axis=1)
                acc = acc * exp_diff[:, None] + tl.dot(
                    exp_s.to(tl.bfloat16), V_tile, out_dtype=tl.float32
                )
                m_i = m_new

                if NO_DIVMOD_2D:
                    kv_dw = kv_dw + STEP_DW
                    c_w = kv_dw >= WU
                    kv_dw = tl.where(c_w, kv_dw - WU, kv_dw)
                    kv_dh = kv_dh + STEP_DH + c_w.to(tl.int32)

    safe_l = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / safe_l[:, None]

    # Recompute output indices at epilogue so q_idx does not occupy int64 VGPRs
    # across the entire inner loop.
    q_idx_out = (
        (t_bid * BT + q_dt) * HW64 + (h_bid * BH + q_dh) * W64 + (w_bid * BW + q_dw)
    )
    tl.store(
        Out_ptr + o_base + q_idx_out[:, None] * o_stride_seq + hd_offs[None, :],
        acc.to(tl.bfloat16),
        mask=q_mask[:, None],
        eviction_policy="evict_first",
    )
