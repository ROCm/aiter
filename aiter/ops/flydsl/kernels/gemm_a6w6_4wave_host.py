# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The 4-wave MXFP6 (A6W6) GEMM as a host for ``GemmHooks`` (``gemm_hooks``).

The main loop is ``gemm_a6w6_4wave``'s: three LDS stages of packed FP6 operands and
their E8M0 scales, ``v_mfma_scale_f32_16x16x128_f8f6f4`` into AGPRs. Here the
activations are the MFMA's first operand, so lane ``l`` of output tile (i, j) holds row
``16 i + 4 (l // 16) + r`` in register ``r``, the layout ``qk_prep_epilogue`` stores.
The weights come in the host layout (``host_weight_order``): B tile ``j`` of a 128-row
group holds the group's rows ``8 k + j``, so lane ``l`` holds column ``8 (l % 16) + j``
and the eight tiles give it eight adjacent columns. The shared epilogue (``StoreCRows``:
bias, Q/K prep, maxima) then applies as it is, to FP32 accumulators the scaled MFMA has
already scaled.

``Columns`` (FP32) is ``StoreCRows``' B_scale: N column scales, which are not applied
here, then N biases with a bias, then the prep's column vectors
(``PrepSpec.pack_columns``).

Modes: one projection (``roles == 1``), plain or with the bias, prep or maxima epilogue,
persistent over ``num_cu`` workgroups or one block per workgroup; or ``roles`` column
blocks of one fused weight (Q/K/V), persistent, with ``hooks`` under the ``gemm_hooks``
contract. A workgroup runs its blocks one after another: a block's prologue starts after
the previous block's epilogue.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from aiter.jit.utils.chip_info import get_cdna4_num_xcds
from aiter.ops.flydsl.kernels.communication_ops_utils import _to_ptr_global
from aiter.ops.flydsl.kernels.gemm_a6w6_4wave import (
    B_OFFSET,
    BLOCK_K,
    BLOCK_M,
    BLOCK_N,
    N_WAVES,
    PADK,
    SCALE_TILE_BYTES,
    SCALES_OFFSET,
    STAGE_BYTES,
    STAGES,
    TILE_BYTES,
    TILE_VMEM,
    WAVE_ROWS,
    WAVE_TILES,
    G2SOperand,
    G2SScales,
    Schedule,
    _lds_load,
    _mfma,
    _read_operand,
)
from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import _xcd_swizzle_any
from aiter.ops.flydsl.kernels.gemm_hooks import GemmHooks
from aiter.ops.flydsl.kernels.kernels_common import (
    ceildiv,
    set_slp_vectorize_once_used,
)
from aiter.ops.flydsl.kernels.qk_prep import PrepSpec
from aiter.ops.flydsl.kernels.qk_prep_epilogue import StoreCRows, mfma_drain


def host_weight_order(n):
    """Row order of a weight with ``n`` rows (a multiple of 128) in the host layout.

    Packed row ``p`` holds row ``order[p]``: 16-row block ``j`` of each 128-row group
    holds the group's rows ``8 k + j``. Pack ``w[host_weight_order(n)]`` with
    ``quant_mxfp6_gemm``.
    """
    import torch

    if n % WAVE_ROWS:
        raise ValueError(f"the host layout needs rows in multiples of 128, got {n}")
    p = torch.arange(n)
    return (p // 128) * 128 + 8 * (p % 16) + (p % 128) // 16


def compile_a6w6_host_4w(
    *,
    K: int,
    xcd_swizzle: int = 8,
    num_cu: int = 0,
    num_xcds: int | None = None,
    mblk_n: int = 2,
    bar: int | None = 8,
    bias: bool = False,
    prep: PrepSpec | None = None,
    amax: bool = False,
    roles: int = 1,
    role_cols: int = 0,
    hooks: GemmHooks | None = None,
    col_amax_offset: int | None = None,
):
    """``K`` is the padded contraction dimension, a positive multiple of 128.

    ``prep`` (a ``PrepSpec``, whose ``bias`` replaces ``bias``) applies the Q/K prep in
    the epilogue, from its FP32 tables (``PrepSpec.pack_tables``), and writes FP32
    ``[2, M, N // 128]`` partials: the row-factor op's per-head partials, then per-head
    maxima of the stored output. ``amax`` (without ``prep``) writes only the
    maxima half. ``roles > 1`` needs ``prep``, ``hooks`` and ``num_cu``: the prep applies
    to every role but the last, C is planar ``[roles, M, role_cols]`` and ``aux`` starts
    with a tile counter per role. ``col_amax_offset`` makes the last role also write FP32
    per-column maxima of its stored output to ``aux`` plus that many bytes,
    ``[M / 128, role_cols]``.
    """
    assert K % BLOCK_K == 0 and K >= BLOCK_K
    if prep is not None:
        if not isinstance(prep, PrepSpec):
            raise TypeError(f"prep must be a PrepSpec or None, got {prep!r}")
        bias = prep.bias
    if num_xcds is None:
        num_xcds = get_cdna4_num_xcds() if xcd_swizzle > 0 else 8
    num_xcds = int(num_xcds)
    persistent = num_cu > 0
    if roles > 1 and (
        prep is None
        or hooks is None
        or not persistent
        or role_cols % BLOCK_N
        or num_cu % num_xcds
        or xcd_swizzle <= 0
    ):
        raise ValueError(
            "roles needs prep, hooks, a persistent grid of a multiple of the XCDs, a "
            "swizzled tile order and role_cols % 256 == 0"
        )
    if col_amax_offset is not None and roles < 2:
        raise ValueError("col_amax_offset needs roles > 1")
    role_hook, first_block_hook, tile_hook = (
        (hooks.role, hooks.first_block, hooks.tile) if hooks else (None, None, None)
    )
    sched = Schedule(mblk_n, bar)
    K_TILES = K // BLOCK_K
    NK_PAD = K_TILES + PADK
    LOADING = K_TILES - STAGES
    TRIPS = max(0, (LOADING - 1) // STAGES)
    STATIC_FROM = 1 + TRIPS * STAGES
    FIRST_BLOCK = first_block_hook is not None

    _kname = (
        f"flydsl_a6w6_host_4w_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_{sched.tag}"
        f"_1x{xcd_swizzle}_k{K}"
        + (f"_p{num_cu}" if persistent else "")
        + (f"_xcd{num_xcds}" if num_xcds != 8 else "")
        + ("_bias" if bias else "")
        + ("_prep" + prep.suffix if prep is not None else "")
        + ("_amax" if amax and prep is None else "")
        + (f"_roles{roles}x{role_cols}" if roles > 1 else "")
        + (f"_hook{hooks.key}" if hooks else "")
        + ("_fb" if FIRST_BLOCK else "")
        + ("_th" if tile_hook is not None else "")
        + (f"_cmax{col_amax_offset}" if col_amax_offset is not None else "")
    )

    @fx.struct
    class SharedStorage:
        stage_0: fx.Array[fx.Int8, STAGE_BYTES, 16]
        stage_1: fx.Array[fx.Int8, STAGE_BYTES, 16]
        stage_2: fx.Array[fx.Int8, STAGE_BYTES, 16]

    @flyc.kernel(name=_kname, known_block_size=[N_WAVES * 64, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        Columns: fx.Tensor,
        Tables: fx.Tensor,
        Partials: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        aux: fx.Int64,
        h0: fx.Int64,
        h1: fx.Int64,
        h2: fx.Int64,
        h3: fx.Int64,
        h4: fx.Int64,
        h5: fx.Int64,
    ):
        m_blocks = ceildiv(c_m, BLOCK_M)
        n_blocks = c_n // BLOCK_N
        hook_args = (h0, h1, h2, h3, h4, h5)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        stage_ptrs = [lds.stage_0.ptr, lds.stage_1.ptr, lds.stage_2.ptr]

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 2
        wave_n = wave_id % 2

        store_c = StoreCRows(
            Columns,
            Columns,
            C,
            c_m,
            c_n,
            None,
            WAVE_TILES,
            WAVE_TILES,
            has_bias=bias,
            prep=(Tables, Partials) if prep is not None else None,
            prep_spec=prep,
            amax_partials=Partials if amax and prep is None else None,
            planar_cols=role_cols if roles > 1 else 0,
            fp32_acc=True,
        )

        def block_coords(block, role):
            if const_expr(roles > 1):
                m, n = _xcd_swizzle_any(
                    m_blocks, role_cols // BLOCK_N, xcd_swizzle, block, num_xcds
                )
                return m, n + role * (role_cols // BLOCK_N)
            if const_expr(xcd_swizzle > 0):
                return _xcd_swizzle_any(
                    m_blocks, n_blocks, xcd_swizzle, block, num_xcds
                )
            return block // n_blocks, block % n_blocks

        def read_scales(stage):
            ptr = stage_ptrs[stage]
            return (
                _lds_load(ptr, SCALES_OFFSET + wave_m * 512 + lane_id * 8, 8),
                _lds_load(
                    ptr,
                    SCALES_OFFSET + SCALE_TILE_BYTES + wave_n * 512 + lane_id * 8,
                    8,
                ),
            )

        def read_a(stage, i):
            return _read_operand(stage_ptrs[stage], 0, wave_m * WAVE_TILES + i, lane_id)

        def read_b(stage, j):
            return _read_operand(
                stage_ptrs[stage], B_OFFSET, wave_n * WAVE_TILES + j, lane_id
            )

        def flat_state(accs, a_ops, b_ops, scales):
            return (
                [accs[i][j] for i in range(WAVE_TILES) for j in range(WAVE_TILES)]
                + list(a_ops)
                + list(b_ops)
                + list(scales)
            )

        def unflat_state(values):
            values = list(values)
            accs = [
                [values.pop(0) for _ in range(WAVE_TILES)] for _ in range(WAVE_TILES)
            ]
            a_ops = [values.pop(0) for _ in range(WAVE_TILES)]
            b_ops = [values.pop(0) for _ in range(WAVE_TILES)]
            scales = (values.pop(0), values.pop(0))
            return accs, a_ops, b_ops, scales

        # The runtime loops stay in the kernel body, where FlyDSL rewrites them; the
        # helpers below hold none. A non-persistent grid runs one block per workgroup.
        grid = fx.Int32(num_cu) if const_expr(persistent) else fx.Int32(fx.grid_dim.x)
        for role in range_constexpr(roles):
            block0 = fx.Int32(fx.block_idx.x)
            if const_expr(roles > 1):
                block_end = m_blocks * (role_cols // BLOCK_N)
            else:
                block_end = m_blocks * n_blocks
            for block, bstate in range(block0, block_end, grid, init=[fx.Int32(0)]):
                block = fx.Int32(block)
                block_m, block_n = block_coords(block, role)
                a_row = fx.Int32(block_m) * NK_PAD
                b_row = fx.Int32(block_n) * NK_PAD
                # The host's SGPR pressure (hook arguments, epilogue buffers) makes LLVM
                # spill SGPRs, so the asm loads take the hazard wait states.
                g2s = (
                    G2SOperand(
                        A, wave_id, stage_ptrs, 0, a_row * TILE_BYTES, lane_id, True
                    ),
                    G2SOperand(
                        B,
                        wave_id,
                        stage_ptrs,
                        B_OFFSET,
                        b_row * TILE_BYTES,
                        lane_id,
                        True,
                    ),
                )
                g2s_scales = G2SScales(
                    A_scale, B_scale, wave_id, stage_ptrs, a_row, b_row, lane_id, True
                )
                row_base = fx.Int32(block_m) * BLOCK_M + wave_m * WAVE_ROWS
                col_base = fx.Int32(block_n) * BLOCK_N + wave_n * WAVE_ROWS

                def tile_vmem(t, stage, g2s=g2s, g2s_scales=g2s_scales):
                    return (
                        g2s[0].ops(t, stage)
                        + g2s[1].ops(t, stage)
                        + g2s_scales.ops(t, stage)
                    )

                def k_tile(
                    accs,
                    a_ops,
                    b_ops,
                    scales,
                    stage,
                    has_next,
                    vmem_wait,
                    prefetch,
                    tile_vmem=tile_vmem,
                ):
                    """One K tile in LDS ``stage``, as in ``gemm_a6w6_4wave``; the last
                    tile (no ``has_next``) only runs its MFMAs."""
                    next_stage = (stage + 1) % STAGES
                    new_a = [None] * WAVE_TILES
                    new_b = [None] * WAVE_TILES
                    new_scales = None
                    sa = (Vec(scales[0])[0], Vec(scales[0])[1])
                    sb = (Vec(scales[1])[0], Vec(scales[1])[1])
                    vmem = [] if prefetch is None else tile_vmem(*prefetch)
                    for idx, (j, i) in enumerate(sched.order):
                        accs[i][j] = _mfma(
                            a_ops[i], b_ops[j], sa[i // 4], sb[j // 4], i, j, accs[i][j]
                        )
                        fx.rocdl.sched_barrier(0)
                        if const_expr(not has_next or idx < sched.bar):
                            continue
                        if const_expr(idx == sched.bar):
                            fx.rocdl.s_waitcnt(vmcnt=vmem_wait, lgkmcnt=0)
                            fx.rocdl.sched_barrier(0)
                            fx.rocdl.s_barrier()
                            fx.rocdl.sched_barrier(0)
                            new_scales = read_scales(next_stage)
                            fx.rocdl.sched_barrier(0)
                        if const_expr(len(vmem) > 0 and idx in sched.vmem_slots):
                            vmem.pop(0)()
                            fx.rocdl.sched_barrier(0)
                        for jj in range_constexpr(WAVE_TILES):
                            if const_expr(
                                new_b[jj] is None and sched.last_b[jj] <= idx
                            ):
                                new_b[jj] = read_b(next_stage, jj)
                                fx.rocdl.sched_barrier(0)
                        for ii in range_constexpr(WAVE_TILES):
                            if const_expr(
                                new_a[ii] is None and sched.last_a[ii] <= idx
                            ):
                                new_a[ii] = read_a(next_stage, ii)
                                fx.rocdl.sched_barrier(0)
                    while const_expr(len(vmem) > 0):
                        vmem.pop(0)()
                        fx.rocdl.sched_barrier(0)
                    return accs, new_a, new_b, new_scales

                def static_step(accs, a_ops, b_ops, scales, t, k_tile=k_tile):
                    return k_tile(
                        accs,
                        a_ops,
                        b_ops,
                        scales,
                        t % STAGES,
                        t + 1 < K_TILES,
                        TILE_VMEM if t + 2 < K_TILES else 0,
                        (t + STAGES, t % STAGES) if t < LOADING else None,
                    )

                def epilogue_loads(row_base=row_base, col_base=col_base):
                    # The bias and norm-weight loads land under the last tile's MFMAs.
                    values = [op() for op in store_c.scale_load_ops(row_base, col_base)]
                    fx.rocdl.sched_barrier(0)
                    return values

                # Prologue: tiles 0, 1 and 2 in flight; wait for tile 0 and read it.
                for t in range_constexpr(STAGES):
                    for op in tile_vmem(t, t):
                        op()
                fx.rocdl.s_waitcnt(vmcnt=(STAGES - 1) * TILE_VMEM)
                fx.rocdl.s_barrier()
                scales = read_scales(0)
                a_ops = [read_a(0, i) for i in range_constexpr(WAVE_TILES)]
                b_ops = [read_b(0, j) for j in range_constexpr(WAVE_TILES)]
                accs = [[None] * WAVE_TILES for _ in range(WAVE_TILES)]

                epilogue_values = epilogue_loads() if K_TILES == 1 else []
                accs, a_ops, b_ops, scales = static_step(accs, a_ops, b_ops, scales, 0)
                if const_expr(TRIPS > 0):
                    for trip, state in range(
                        0, TRIPS, 1, init=flat_state(accs, a_ops, b_ops, scales)
                    ):
                        accs, a_ops, b_ops, scales = unflat_state(state)
                        t0 = fx.Int32(trip) * STAGES + 1
                        for u in range_constexpr(STAGES):
                            accs, a_ops, b_ops, scales = k_tile(
                                accs,
                                a_ops,
                                b_ops,
                                scales,
                                (1 + u) % STAGES,
                                True,
                                TILE_VMEM,
                                (t0 + u + STAGES, (1 + u) % STAGES),
                            )
                        results = yield flat_state(accs, a_ops, b_ops, scales)
                    accs, a_ops, b_ops, scales = unflat_state(results)
                for t in range_constexpr(STATIC_FROM, K_TILES):
                    if const_expr(t == K_TILES - 1):
                        epilogue_values = epilogue_loads()
                    accs, a_ops, b_ops, scales = static_step(
                        accs, a_ops, b_ops, scales, t
                    )
                # The epilogue's loads are asm, so LLVM does not wait for them; this
                # also drains the guard-tile loads below three K tiles.
                fx.rocdl.s_waitcnt(vmcnt=0)
                col_max = None
                if const_expr(col_amax_offset is not None and role == roles - 1):
                    col_max = aux + col_amax_offset
                store_c.store_tiles(
                    mfma_drain(accs),
                    row_base,
                    col_base,
                    epilogue_values,
                    prep=(role < roles - 1) if roles > 1 else None,
                    col_max=col_max,
                )
                # Every wave's stores land, and every wave is done with LDS, before the
                # tile is counted and the next block's loads refill the stages.
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.barrier()
                if const_expr(roles > 1):
                    if const_expr(tile_hook is not None):
                        tile_hook(role, aux, hook_args, c_m, row_base, col_base)
                    if fx.thread_idx.x == 0:
                        fx.atomic_add(
                            _to_ptr_global(aux + role * 4, fx.Int32),
                            fx.Int32(1),
                            syncscope=fx.rocdl.SyncScope.Agent,
                            ordering=fx.AtomicOrdering.Release,
                        )
                    # A trace-time branch around a runtime one; they do not merge.
                    if const_expr(FIRST_BLOCK and role > 0):  # noqa: SIM102
                        if block == block0:
                            first_block_hook(role, aux, hook_args, c_m)
                _ = yield [bstate[0]]
            if const_expr(role_hook is not None):
                role_hook(role, aux, hook_args, c_m)

    value_attrs = {
        "rocdl.waves_per_eu": 1,
        "rocdl.flat_work_group_size": f"{N_WAVES * 64},{N_WAVES * 64}",
    }

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        Columns: fx.Tensor,
        Tables: fx.Tensor,
        Partials: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = num_cu if persistent else ceildiv(c_m, BLOCK_M) * (c_n // BLOCK_N)
        zero = fx.Int64(0)
        kernel_gemm(
            A,
            B,
            C,
            A_scale,
            B_scale,
            Columns,
            Tables,
            Partials,
            c_m,
            c_n,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            value_attrs=value_attrs,
        ).launch(grid=(grid_x, 1, 1), block=(N_WAVES * 64, 1, 1), stream=stream)

    @flyc.jit
    def launch_gemm_roles(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        Columns: fx.Tensor,
        Tables: fx.Tensor,
        Partials: fx.Tensor,
        aux: fx.Int64,
        h0: fx.Int64,
        h1: fx.Int64,
        h2: fx.Int64,
        h3: fx.Int64,
        h4: fx.Int64,
        h5: fx.Int64,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_gemm(
            A,
            B,
            C,
            A_scale,
            B_scale,
            Columns,
            Tables,
            Partials,
            c_m,
            c_n,
            aux,
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            value_attrs=value_attrs,
        ).launch(grid=(num_cu, 1, 1), block=(N_WAVES * 64, 1, 1), stream=stream)

    launcher = launch_gemm_roles if roles > 1 else launch_gemm
    return set_slp_vectorize_once_used(launcher, False)
