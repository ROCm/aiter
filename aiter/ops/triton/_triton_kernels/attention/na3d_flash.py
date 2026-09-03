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
- Autotuned over BLOCK_Q (16, 32), num_stages (2, 3, 4), num_warps (4, 8).
"""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# Autotune configs.
# BLOCK_KV = next_pow2(BLOCK_Q + KW - 1) covers the union of all BLOCK_Q
# queries' W windows in one chunk:
#   BLOCK_Q=16  ->  BLOCK_KV=32
#   BLOCK_Q=32  ->  BLOCK_KV=64
# The W >= BLOCK_Q constraint is enforced by the pruner below.
_CONFIGS = [
    triton.Config({"BLOCK_Q": 16, "BLOCK_KV": 32, "num_stages": 2}, num_warps=4),
    triton.Config({"BLOCK_Q": 16, "BLOCK_KV": 32, "num_stages": 2}, num_warps=8),
    triton.Config({"BLOCK_Q": 16, "BLOCK_KV": 32, "num_stages": 3}, num_warps=4),
    triton.Config({"BLOCK_Q": 16, "BLOCK_KV": 32, "num_stages": 3}, num_warps=8),
    triton.Config({"BLOCK_Q": 16, "BLOCK_KV": 32, "num_stages": 4}, num_warps=4),
    triton.Config({"BLOCK_Q": 16, "BLOCK_KV": 32, "num_stages": 4}, num_warps=8),
    # BLOCK_Q=32 halves program count but doubles BLOCK_KV (more masked compute).
    triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 64, "num_stages": 2}, num_warps=4),
    triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 64, "num_stages": 2}, num_warps=8),
]


def _prune_configs(configs, named_args, **kwargs):
    """Remove configs where BLOCK_Q > W or BLOCK_KV is too small for KW.

    Performance constraint: BLOCK_Q > W would leave most of every block masked,
    wasting compute. Correctness also requires BLOCK_KV >= (BLOCK_Q + KW - 1)
    so the KV tile covers the union of all BLOCK_Q query windows.
    """
    W = named_args["W"]
    KW = kwargs.get("KW")  # constexpr in kwargs
    return [
        c
        for c in configs
        if c.kwargs["BLOCK_Q"] <= W
        and (KW is None or c.kwargs["BLOCK_KV"] >= c.kwargs["BLOCK_Q"] + KW - 1)
    ]


_na3d_flash_fwd_repr = make_kernel_repr(
    "_na3d_flash_fwd",
    ["BLOCK_Q", "BLOCK_KV", "KT", "KH", "KW", "HD"],
)


@triton.autotune(
    configs=_CONFIGS,
    key=["KT", "KH", "KW", "HD", "W"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit(repr=_na3d_flash_fwd_repr)
def _na3d_flash_fwd(
    Q_ptr,
    K_ptr,
    V_ptr,
    Out_ptr,
    stride_bnh,
    stride_seq,
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

    Each program is assigned to exactly one (t, h) row via decomposition of
    pid_q into (row_idx, w_block_idx).  This guarantees a shared (t, h) window
    start regardless of whether W is divisible by BLOCK_Q.
    """
    pid_q = tl.program_id(0)
    pid_bnh = tl.program_id(1)
    H64 = H.to(tl.int64)
    W64 = W.to(tl.int64)
    HW = H64 * W64

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

    # Cast to int64: pid_bnh (int32) x stride_bnh can exceed int32 range.
    base = pid_bnh.to(tl.int64) * stride_bnh
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
