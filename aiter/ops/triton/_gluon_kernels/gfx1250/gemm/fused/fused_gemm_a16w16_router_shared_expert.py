# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# The MoE router GEMM and the whole bf16 shared expert of a decode batch in one
# persistent kernel:
#   logits = x @ Wr^T                              [M, NR]
#   h      = silu(x @ Wg^T) * (x @ Wu^T)           [M, I]  (scratch)
#   shared = h @ Wd^T                              [M, ND]
# The programs walk one tile list in order: the gate / up tiles (B rows gate /
# up interleaved, as gemm_a16w16_gated reads them), then the router tiles, then
# the down tiles. A gate / up tile counts itself done in the counter of its
# M block (one device-scope add per wave once its h stores, device scope, are
# complete; no agent-scope fence, which writes back the whole L2), and a down
# tile polls that counter before it reads its h rows (device scope). A down
# tile holds its whole K (= I) in LDS: its Wd loads go out before the poll.
# The router weight may carry zero rows past NR (a multiple of 16 rows keeps
# the last router tile's TDM loads fast); only NR logits are stored. Each program does all of its earlier tiles
# before a later one, so no program waits on a tile that waits; with every
# program resident (the wrapper caps the grid) the kernel cannot deadlock.
# The polls time out and count the timeout in ws[0] instead of hanging.
# Every tile runs the K chain of the a16w16 GEMM in K order with one fp32
# accumulator, so all three outputs are bitwise those of the separate GEMMs.

import triton.experimental.gluon.language as gl
from triton.experimental import gluon

from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (
    _compute_bound_accumulate,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_router_shared_expert_repr = make_kernel_repr(
    "_gemm_a16w16_router_shared_expert_kernel",
    [
        "BLOCK_M",
        "BLOCK_N1",
        "BLOCK_K1",
        "NUM_BUFFERS1",
        "BLOCK_N2",
        "BLOCK_K2",
        "NUM_PROGRAMS",
    ],
)

# Workspace: int32 ws[0] timeouts, ws[2 + mb] the wave arrivals of M block
# mb's gate / up tiles, ws[2 + WS_MAX_MB + mb] the down tiles that read it.
WS_HEADER = gl.constexpr(2)
WS_MAX_MB = gl.constexpr(1024)


@gluon.jit
def _wave_arrive(ptr, NUM_WARPS: gl.constexpr):
    """One device-scope add to *ptr per wave, from lane 0, once the wave's own
    stores are complete. No agent-scope fence: on gfx1250 that writes back
    the whole L2. The h stores carry device scope instead."""
    gl.inline_asm("s_wait_storecnt 0x0", "~{memory}", [], [], is_pure=False)
    L: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [NUM_WARPS, 1], [1, 0])
    lanes = gl.arange(0, 32, layout=gl.SliceLayout(0, L))
    waves = gl.arange(0, NUM_WARPS, layout=gl.SliceLayout(1, L))
    zero = waves[:, None] * 0 + lanes[None, :] * 0
    gl.atomic_add(
        ptr + zero,
        zero + 1,
        mask=(zero + lanes[None, :]) == 0,
        sem="relaxed",
        scope="gpu",
    )


@gluon.jit
def _store_tile(
    vals, c_ptr, M, N, stride_cm, row0, col0, LAYOUT: gl.constexpr, CACHE: gl.constexpr
):
    """Masked bf16 store of a [BM, BN] tile in LAYOUT at (row0, col0)."""
    BM: gl.constexpr = vals.shape[0]
    BN: gl.constexpr = vals.shape[1]
    rows = gl.arange(0, BM, layout=gl.SliceLayout(1, LAYOUT))
    cols = gl.arange(0, BN, layout=gl.SliceLayout(0, LAYOUT))
    base = c_ptr + row0.to(gl.int64) * stride_cm + col0
    offs = rows[:, None] * stride_cm + cols[None, :]
    mask = (rows[:, None] < M - row0) & (cols[None, :] < N - col0)
    gl.amd.gfx1250.buffer_store(
        vals.to(c_ptr.type.element_ty), base, offs, mask=mask, cache=CACHE
    )


@gluon.jit
def _store_tile_own_layout(vals, c_ptr, M, N, stride_cm, row0, col0):
    """`_store_tile` in the layout `vals` already has."""
    LAYOUT: gl.constexpr = vals.type.layout
    _store_tile(vals, c_ptr, M, N, stride_cm, row0, col0, LAYOUT, "")


@gluon.jit
def _gated_silu_mul_store_lds(
    acc,
    h_ptr,
    M,
    stride_hm,
    row0,
    col0,
    ROWS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """h[row0:, col0:] = silu(gate) * up of a gate / up interleaved fp32 tile
    [BM, BN], through LDS: the tile goes to LDS as bf16 (gate and up round
    to bf16 before the SiLU, as in gemm_a16w16_gated), then ROWS rows at a
    time come back in a layout that holds each gate / up pair in one lane.
    The accumulator dies at the LDS store instead of living through the
    epilogue (a 256 x 256 tile otherwise spills)."""
    BM: gl.constexpr = acc.shape[0]
    BN: gl.constexpr = acc.shape[1]
    HALF: gl.constexpr = BN // 2
    SHARED_C: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BN, 8]], [BM, BN], [1, 0]
    )
    c_buffer = gl.allocate_shared_memory(
        h_ptr.type.element_ty, shape=[BM, BN], layout=SHARED_C
    )
    c_buffer.store(acc.to(h_ptr.type.element_ty))
    gl.barrier()
    TPW_N: gl.constexpr = min(32, BN // 8)
    LOAD: gl.constexpr = gl.BlockedLayout(
        [1, 8], [32 // TPW_N, TPW_N], [NUM_WARPS, 1], [1, 0]
    )
    for r in gl.static_range(BM // ROWS):
        v = c_buffer.slice(r * ROWS, ROWS, 0).load(LOAD).to(gl.float32)
        gate, up = gl.split(gl.reshape(v, [ROWS, HALF, 2]))
        hv = gate / (1.0 + gl.exp(-gate)) * up
        _store_tile_own_layout(
            hv, h_ptr, M, col0 + HALF, stride_hm, row0 + r * ROWS, col0
        )


@gluon.jit
def _tdm_store_tile(vals, c_ptr, M, N, stride_cm, row0, col0):
    """bf16 store of a [BM, BN] tile at (row0, col0) through LDS by TDM: whole
    rows of any stride (the 257-wide logits), clipped at (M, N)."""
    BM: gl.constexpr = vals.shape[0]
    BN: gl.constexpr = vals.shape[1]
    SHARED_C: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BN, 8]], [BM, BN], [1, 0]
    )
    c_buffer = gl.allocate_shared_memory(
        c_ptr.type.element_ty, shape=[BM, BN], layout=SHARED_C
    )
    c_buffer.store(vals.to(c_ptr.type.element_ty))
    gl.barrier()
    c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, 1),
        block_shape=(BM, BN),
        layout=SHARED_C,
    )
    gl.amd.gfx1250.tdm.async_store(c_desc, [row0, col0], c_buffer)
    gl.amd.gfx1250.tdm.async_wait(0)


@gluon.jit
def _down_tile(
    h_ptr,
    wd_ptr,
    out_ptr,
    cnt_ptr,
    use_ptr,
    ws_ptr,
    M,
    stride_hm,
    stride_om,
    mb,
    nb,
    expected,
    KD: gl.constexpr,
    ND: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SHARED_A: gl.constexpr,
    SHARED_B: gl.constexpr,
    WMMA: gl.constexpr,
    OP_A: gl.constexpr,
    OP_B: gl.constexpr,
    N_D: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    H_CACHE: gl.constexpr,
    TIMEOUT_NS: gl.constexpr,
):
    """Tile (mb, nb) of h @ Wd^T with the whole K (= I) resident in LDS: the
    Wd loads go out before the poll for h, the h loads right after it. The
    K chain is that of the a16w16 GEMM: one fp32 accumulator, K in order."""
    NK: gl.constexpr = KD // BLOCK_K
    a_buffer = gl.allocate_shared_memory(
        h_ptr.type.element_ty, shape=[NK, BLOCK_M, BLOCK_K], layout=SHARED_A
    )
    b_buffer = gl.allocate_shared_memory(
        wd_ptr.type.element_ty, shape=[NK, BLOCK_N, BLOCK_K], layout=SHARED_B
    )
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=wd_ptr + nb * BLOCK_N * KD,
        shape=(ND - nb * BLOCK_N, KD),
        strides=(KD, 1),
        block_shape=(BLOCK_N, BLOCK_K),
        layout=SHARED_B,
    )
    for k in gl.static_range(NK):
        gl.amd.gfx1250.tdm.async_load(b_desc, [0, k * BLOCK_K], b_buffer.index(k))
    ok = gl.atomic_poll(
        cnt_ptr + mb, expected, sem="relaxed", scope="gpu", timeout_ns=TIMEOUT_NS
    )
    if not ok:
        gl.atomic_add(ws_ptr, 1, sem="relaxed", scope="gpu")
    # Count this consumer of h block mb. The last one clears both counters
    # for the next launch; its add returns while the tile computes.
    L: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [NUM_WARPS, 1], [1, 0])
    lanes = gl.arange(0, 32, layout=gl.SliceLayout(0, L))
    waves = gl.arange(0, NUM_WARPS, layout=gl.SliceLayout(1, L))
    lead = (waves[:, None] + lanes[None, :]) == 0
    zero = waves[:, None] * 0 + lanes[None, :] * 0
    used = gl.atomic_add(
        use_ptr + mb + zero, zero + 1, mask=lead, sem="relaxed", scope="gpu"
    )
    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=h_ptr + mb * BLOCK_M * stride_hm,
        shape=(M - mb * BLOCK_M, KD),
        strides=(stride_hm, 1),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=SHARED_A,
    )
    for k in gl.static_range(NK):
        gl.amd.gfx1250.tdm.async_load(
            a_desc, [0, k * BLOCK_K], a_buffer.index(k), cache_modifier=H_CACHE
        )
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA)
    for k in gl.static_range(NK):
        gl.amd.gfx1250.tdm.async_wait(NK - 1 - k)
        cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_buffer.index(k), OP_A)
        cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index(k).permute([1, 0]), OP_B
        )
        acc = gl.amd.gfx1250.wmma(cur_a, cur_b, acc)
    _store_tile(acc, out_ptr, M, ND, stride_om, mb * BLOCK_M, nb * BLOCK_N, WMMA, "")
    last = lead & (used == N_D - 1)
    gl.atomic_xchg(cnt_ptr + mb + zero, zero, mask=last, sem="relaxed", scope="gpu")
    gl.atomic_xchg(use_ptr + mb + zero, zero, mask=last, sem="relaxed", scope="gpu")


@gluon.jit(repr=_router_shared_expert_repr)
def _gemm_a16w16_router_shared_expert_kernel(
    x_ptr,
    wr_ptr,
    wgu_ptr,
    wd_ptr,
    logits_ptr,
    h_ptr,
    out_ptr,
    ws_ptr,
    M,
    stride_xm,
    stride_lm,
    stride_hm,
    stride_om,
    K: gl.constexpr,
    NR: gl.constexpr,
    NR_ROWS: gl.constexpr,
    NGU: gl.constexpr,
    ND: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N1: gl.constexpr,
    BLOCK_K1: gl.constexpr,
    NUM_BUFFERS1: gl.constexpr,
    BLOCK_N2: gl.constexpr,
    BLOCK_K2: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    LAYOUT: gl.constexpr,
    SHARED_A1: gl.constexpr,
    SHARED_B1: gl.constexpr,
    SHARED_A2: gl.constexpr,
    SHARED_B2: gl.constexpr,
    WMMA1: gl.constexpr,
    OP_A1: gl.constexpr,
    OP_B1: gl.constexpr,
    WMMA2: gl.constexpr,
    OP_A2: gl.constexpr,
    OP_B2: gl.constexpr,
    STORE1: gl.constexpr,
    TIMEOUT_NS: gl.constexpr,
    H_CACHE: gl.constexpr,
):
    KD: gl.constexpr = NGU // 2
    N_GU: gl.constexpr = NGU // BLOCK_N1
    N_R: gl.constexpr = (NR + BLOCK_N1 - 1) // BLOCK_N1
    N_D: gl.constexpr = ND // BLOCK_N2
    HALF: gl.constexpr = BLOCK_N1 // 2
    pid = gl.program_id(axis=0)
    num_mb = gl.cdiv(M, BLOCK_M)
    t_gu = num_mb * N_GU
    t_1 = t_gu + num_mb * N_R
    t_all = t_1 + num_mb * N_D
    cnt_ptr = ws_ptr + WS_HEADER

    for t in range(pid, t_all, NUM_PROGRAMS):
        if t < t_gu:
            mb = t // N_GU
            nb = t % N_GU
            acc = _compute_bound_accumulate(
                x_ptr,
                wgu_ptr,
                M,
                NGU,
                K,
                stride_xm,
                1,
                1,
                K,
                mb,
                nb,
                BLOCK_M,
                BLOCK_N1,
                BLOCK_K1,
                NUM_BUFFERS1,
                LAYOUT,
                SHARED_A1,
                SHARED_B1,
                WMMA1,
                OP_A1,
                OP_B1,
            )
            # As gemm_a16w16_gated: gate and up round to bf16 before the SiLU.
            gate, up = gl.split(gl.reshape(acc, [BLOCK_M, HALF, 2]))
            gate = gate.to(h_ptr.type.element_ty).to(gl.float32)
            up = up.to(h_ptr.type.element_ty).to(gl.float32)
            hv = gate / (1.0 + gl.exp(-gate)) * up
            hv = gl.convert_layout(hv, STORE1)
            _store_tile(
                hv, h_ptr, M, KD, stride_hm, mb * BLOCK_M, nb * HALF, STORE1, H_CACHE
            )
            _wave_arrive(cnt_ptr + mb, NUM_WARPS)
        elif t < t_1:
            u = t - t_gu
            mb = u // N_R
            nb = u % N_R
            acc = _compute_bound_accumulate(
                x_ptr,
                wr_ptr,
                M,
                NR_ROWS,
                K,
                stride_xm,
                1,
                1,
                K,
                mb,
                nb,
                BLOCK_M,
                BLOCK_N1,
                BLOCK_K1,
                NUM_BUFFERS1,
                LAYOUT,
                SHARED_A1,
                SHARED_B1,
                WMMA1,
                OP_A1,
                OP_B1,
            )
            _tdm_store_tile(
                acc, logits_ptr, M, NR, stride_lm, mb * BLOCK_M, nb * BLOCK_N1
            )
        else:
            u = t - t_1
            _down_tile(
                h_ptr,
                wd_ptr,
                out_ptr,
                cnt_ptr,
                cnt_ptr + WS_MAX_MB,
                ws_ptr,
                M,
                stride_hm,
                stride_om,
                u // N_D,
                u % N_D,
                N_GU * NUM_WARPS,
                KD,
                ND,
                BLOCK_M,
                BLOCK_N2,
                BLOCK_K2,
                SHARED_A2,
                SHARED_B2,
                WMMA2,
                OP_A2,
                OP_B2,
                N_D,
                NUM_WARPS,
                H_CACHE,
                TIMEOUT_NS,
            )


# ---------------------------------------------------------------------------
# Prefill: the router GEMM and the shared expert's gate_up GEMM as one
# persistent GEMM over the (logically) concatenated weight [gate_up; router]:
#   h      = silu(x @ Wg^T) * (x @ Wu^T)           [M, I]
#   logits = x @ Wr^T                              [M, NR]
# One launch, and the router tiles fill the rounds the gate_up tiles leave
# (the separate router GEMM is a wave of its own). The down GEMM stays
# separate. Three tile classes in one list, largest first:
#   G  gate_up tiles, BLOCK_M1 x BLOCK_N1, SiLU-and-mul epilogue into h;
#   R  router tiles over the first NR_MAIN rows, BLOCK_M2 x BLOCK_N2, bf16
#      logits;
#   T  the router rows past NR_MAIN (the shared-gate logit, zero rows up to
#      NR_ROWS), BLOCK_M2 x BLOCK_NT.
# Within a class the M blocks run fastest, so the N tiles of an M block land
# on one XCD (programs go to XCDs round robin) and share its L2 copy of x.
# Every tile runs the a16w16 K chain (one fp32 accumulator, K in order), so
# h and the logits are bitwise those of gemm_a16w16_gated and the router GEMM.
# ---------------------------------------------------------------------------

_router_gate_up_repr = make_kernel_repr(
    "_gemm_a16w16_router_gate_up_kernel",
    [
        "BLOCK_M1",
        "BLOCK_N1",
        "BLOCK_K1",
        "NUM_BUFFERS1",
        "BLOCK_M2",
        "BLOCK_N2",
        "BLOCK_K2",
        "NUM_BUFFERS2",
        "NUM_PROGRAMS",
    ],
)


@gluon.jit(repr=_router_gate_up_repr)
def _gemm_a16w16_router_gate_up_kernel(
    x_ptr,
    wr_ptr,
    wgu_ptr,
    logits_ptr,
    h_ptr,
    M,
    stride_xm,
    stride_lm,
    stride_hm,
    K: gl.constexpr,
    NR: gl.constexpr,
    NR_MAIN: gl.constexpr,
    NR_ROWS: gl.constexpr,
    NGU: gl.constexpr,
    BLOCK_M1: gl.constexpr,
    BLOCK_N1: gl.constexpr,
    BLOCK_K1: gl.constexpr,
    NUM_BUFFERS1: gl.constexpr,
    BLOCK_M2: gl.constexpr,
    BLOCK_N2: gl.constexpr,
    BLOCK_K2: gl.constexpr,
    NUM_BUFFERS2: gl.constexpr,
    BLOCK_NT: gl.constexpr,
    NUM_PROGRAMS: gl.constexpr,
    H_ROWS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SHARED_A1: gl.constexpr,
    SHARED_B1: gl.constexpr,
    WMMA1: gl.constexpr,
    OP_A1: gl.constexpr,
    OP_B1: gl.constexpr,
    SHARED_A2: gl.constexpr,
    SHARED_B2: gl.constexpr,
    WMMA2: gl.constexpr,
    OP_A2: gl.constexpr,
    OP_B2: gl.constexpr,
    SHARED_BT: gl.constexpr,
    WMMAT: gl.constexpr,
    OP_AT: gl.constexpr,
    OP_BT: gl.constexpr,
):
    N_G: gl.constexpr = NGU // BLOCK_N1
    N_R: gl.constexpr = NR_MAIN // BLOCK_N2
    pid = gl.program_id(axis=0)
    num_mb1 = gl.cdiv(M, BLOCK_M1)
    num_mb2 = gl.cdiv(M, BLOCK_M2)
    t_g = num_mb1 * N_G
    t_r = t_g + num_mb2 * N_R
    t_all = t_r + num_mb2
    wt_ptr = wr_ptr + NR_MAIN * K

    for t in range(pid, t_all, NUM_PROGRAMS):
        if t < t_g:
            mb = t % num_mb1
            nb = t // num_mb1
            acc_g = _compute_bound_accumulate(
                x_ptr,
                wgu_ptr,
                M,
                NGU,
                K,
                stride_xm,
                1,
                1,
                K,
                mb,
                nb,
                BLOCK_M1,
                BLOCK_N1,
                BLOCK_K1,
                NUM_BUFFERS1,
                "TN",
                SHARED_A1,
                SHARED_B1,
                WMMA1,
                OP_A1,
                OP_B1,
            )
            _gated_silu_mul_store_lds(
                acc_g,
                h_ptr,
                M,
                stride_hm,
                mb * BLOCK_M1,
                nb * (BLOCK_N1 // 2),
                H_ROWS,
                NUM_WARPS,
            )
        elif t < t_r:
            u = t - t_g
            mb = u % num_mb2
            nb = u // num_mb2
            acc_r = _compute_bound_accumulate(
                x_ptr,
                wr_ptr,
                M,
                NR_MAIN,
                K,
                stride_xm,
                1,
                1,
                K,
                mb,
                nb,
                BLOCK_M2,
                BLOCK_N2,
                BLOCK_K2,
                NUM_BUFFERS2,
                "TN",
                SHARED_A2,
                SHARED_B2,
                WMMA2,
                OP_A2,
                OP_B2,
            )
            _tdm_store_tile(
                acc_r, logits_ptr, M, NR, stride_lm, mb * BLOCK_M2, nb * BLOCK_N2
            )
        else:
            mb = t - t_r
            acc_t = _compute_bound_accumulate(
                x_ptr,
                wt_ptr,
                M,
                NR_ROWS - NR_MAIN,
                K,
                stride_xm,
                1,
                1,
                K,
                mb,
                0,
                BLOCK_M2,
                BLOCK_NT,
                BLOCK_K2,
                NUM_BUFFERS2,
                "TN",
                SHARED_A2,
                SHARED_BT,
                WMMAT,
                OP_AT,
                OP_BT,
            )
            _store_tile(
                acc_t, logits_ptr, M, NR, stride_lm, mb * BLOCK_M2, NR_MAIN, WMMAT, ""
            )
