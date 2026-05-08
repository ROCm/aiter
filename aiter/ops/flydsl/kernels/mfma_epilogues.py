"""Reusable epilogue helpers for MFMA 16x16-based kernels.

This module provides:

- `mfma_epilog(...)`
  A single entrypoint that dispatches to either the default row-epilogue or the
  LDS CShuffle epilogue based on input parameters.

- `default_epilog(...)` (implementation helper)
  A lightweight row-iterator for the common MFMA accumulator-to-output mapping
  (mi in [0,m_repeat), ii in [0,4), row = bx_m + mi*16 + lane_div_16*4 + ii).
  The caller supplies `body_row(...)` that performs the per-row epilogue work
  (e.g. loads scales once, loops over ni, stores).

- `c_shuffle_epilog(...)` (implementation helper)
  A LDS CShuffle epilogue skeleton:
    1) call `write_row_to_lds(...)` for each MFMA output row to populate `lds_out`
       in row-major [tile_m, tile_n] order
    2) barrier
    3) remap threads into (MLane, NLane) = (8,32) and read half2 from LDS,
       then call `store_pair(...)` to emit the final global store/atomic.

These helpers are intentionally *dialect-agnostic*: callers pass the dialect
modules (`arith`, `vector`, `gpu`) and the `range_constexpr` iterator.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

from flydsl._mlir import ir
import flydsl.expr as fx
from flydsl.expr.typing import T


@contextmanager
def _if_then(if_op, scf):
    """Compat helper for SCF IfOp then-region across old/new Python APIs."""
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


def default_epilog(
    *,
    arith,
    range_constexpr,
    m_repeat: int,
    lane_div_16,
    bx_m,
    body_row: Callable,
):
    """Iterate the standard MFMA 16x16 row mapping and call `body_row(...)`.

    The mapping matches the common MFMA fragment layout used across kernels in this repo.

    Args:
      arith: flydsl arith ext module.
      range_constexpr: compile-time unrolled range helper.
      m_repeat: tile_m // 16 (python int).
      lane_div_16: index Value (0..3).
      bx_m: base row (index Value). For MoE, this is the base sorted-row for the tile.
      body_row: callback invoked as:
        body_row(mi=<int>, ii=<int>, row_in_tile=<index>, row=<index>)
    """
    bx_m_v = bx_m
    lane_div_16_mul4 = lane_div_16 * 4
    ii_idx_list = [fx.Index(ii) for ii in range(4)]

    for mi in range_constexpr(m_repeat):
        mi_base = arith.constant(mi * 16, index=True)
        for ii in range_constexpr(4):
            row_off = lane_div_16_mul4 + ii_idx_list[ii]
            row_in_tile = mi_base + row_off
            row = bx_m_v + row_in_tile
            body_row(mi=mi, ii=ii, row_in_tile=row_in_tile, row=row)


def c_shuffle_epilog(
    *,
    arith,
    vector,
    gpu,
    scf=None,
    range_constexpr,
    # Tile params
    tile_m: int,
    tile_n: int,
    e_vec: int = 2,
    cshuffle_nlane: int = 32,
    block_size: int = 256,
    m_repeat: int,
    num_acc_n: int,
    # Thread mapping inputs
    tx,
    lane_div_16,
    lane_mod_16,
    bx_m,
    by_n,
    n_tile_base,
    # LDS buffer (f16 view, row-major [tile_m, tile_n] flattened)
    lds_out,
    lane_div_32=None,
    lane_mod_32=None,
    mfma_m: int = 16,
    # Element type for LDS loads (defaults to f16). Pass bf16 to support bf16 epilogues.
    frag_elem_type: ir.Type | None = None,
    # Callbacks
    write_row_to_lds: Callable,
    precompute_row: Callable | None = None,
    store_pair: Callable,
    # When True, skip the initial gpu.barrier() before write_row_to_lds.
    # Useful when the caller already issued a barrier right before this call.
    skip_initial_barrier: bool = False,
    # When True and n_reps_shuffle > 1, interleave the LDS column layout so that
    # each CShuffle reader thread's multiple nr fragments are adjacent. This lets
    # the read side use a single wider ds_read instead of multiple narrow reads.
    interleave_n_reps: bool = False,
    # When True, replace per-row branching in Phase 3 with exec masking plus
    # s_setvskip inline asm. Requires the caller to pass the llvm dialect module.
    use_vskip: bool = False,
    llvm=None,
    # Keep the newer three-phase CShuffle read scheduling opt-in. The default
    # streaming path is still better for int8 MoE kernels.
    batch_cshuffle_reads: bool = False,
):
    """LDS CShuffle epilogue skeleton.

    Call pattern:
      - `write_row_to_lds(...)` is called once per MFMA row produced by this thread.
        It is responsible for writing all ni columns for that row into `lds_out`.
      - `store_pair(...)` is called for each (row_local, col_pair0) half2 after shuffle.

    `store_pair` can implement either global stores or atomics.
    """
    if int(block_size) <= 0 or (int(block_size) % int(cshuffle_nlane)) != 0:
        raise ValueError(
            f"block_size ({block_size}) must be divisible by cshuffle_nlane ({cshuffle_nlane})"
        )
    cshuffle_mlane = int(block_size) // int(cshuffle_nlane)
    if (int(tile_m) % cshuffle_mlane) != 0:
        raise ValueError(
            f"tile_m must be divisible by CShuffleMLane ({cshuffle_mlane}), got tile_m={tile_m}"
        )
    if int(e_vec) <= 0:
        raise ValueError(f"e_vec must be positive, got {e_vec}")
    if (int(tile_n) % (int(cshuffle_nlane) * int(e_vec))) != 0:
        raise ValueError(
            f"tile_n must be divisible by (CShuffleNLane*EVec) = {cshuffle_nlane*e_vec}, got tile_n={tile_n}"
        )

    # ---------------- Step 1: write C tile to LDS ----------------
    CShuffleNLane = int(cshuffle_nlane)
    CShuffleMLane = int(cshuffle_mlane)
    EVec = int(e_vec)
    n_reps_shuffle = int(tile_n) // (CShuffleNLane * EVec)

    _do_interleave = interleave_n_reps and n_reps_shuffle > 1

    tile_n_idx = arith.constant(int(tile_n), index=True)
    n_tile_base_v = n_tile_base
    write_lane_mod = lane_mod_32 if int(mfma_m) == 32 else lane_mod_16
    write_lane_div = lane_div_32 if int(mfma_m) == 32 else lane_div_16
    if write_lane_mod is None or write_lane_div is None:
        raise ValueError("mfma_m=32 requires lane_div_32 and lane_mod_32")
    col_base_local = n_tile_base_v + write_lane_mod  # index within [0,tile_n)

    _col_remap = None
    if _do_interleave:
        _nr_step = CShuffleNLane * EVec
        _wide_evec = n_reps_shuffle * EVec
        _c_nr_step = arith.constant(_nr_step, index=True)
        _c_evec_remap = arith.constant(EVec, index=True)
        _c_wide_evec = arith.constant(_wide_evec, index=True)

        def _col_remap(col):
            nr = col // _c_nr_step
            local_c = col % _c_nr_step
            n_lane_col = local_c // _c_evec_remap
            evec_v = local_c % _c_evec_remap
            return n_lane_col * _c_wide_evec + nr * _c_evec_remap + evec_v

    def _write_row(mi: int, ii: int, row_in_tile, row):
        row_base_lds = row_in_tile * tile_n_idx
        _extra = {}
        if _col_remap is not None:
            _extra["lds_col_remap"] = _col_remap
        write_row_to_lds(
            mi=mi,
            ii=ii,
            row_in_tile=row_in_tile,
            row=row,
            row_base_lds=row_base_lds,
            col_base_local=col_base_local,
            num_acc_n=num_acc_n,
            lds_out=lds_out,
            **_extra,
        )

    if not skip_initial_barrier:
        gpu.barrier()
    if int(mfma_m) == 32:
        # CK row-major 32x32 MFMA mapping:
        # row = mi*32 + lane_div32*4 + (ii//4)*8 + (ii%4), ii in [0,16).
        lane_div_32_mul4 = write_lane_div * arith.index(4)
        for mi in range_constexpr(m_repeat):
            mi_base = arith.constant(mi * 32, index=True)
            for ii in range_constexpr(16):
                row_off = lane_div_32_mul4 + arith.constant(
                    (ii // 4) * 8 + (ii % 4), index=True
                )
                row_in_tile = mi_base + row_off
                row = bx_m + row_in_tile
                _write_row(mi, ii, row_in_tile, row)
    else:
        default_epilog(
            arith=arith,
            range_constexpr=range_constexpr,
            m_repeat=m_repeat,
            lane_div_16=lane_div_16,
            bx_m=bx_m,
            body_row=_write_row,
        )

    # Ensure all LDS writes are visible before the shuffle-read.
    gpu.barrier()

    # ---------------- Step 2: shuffle mapping + half2 store/atomic ----------------
    m_reps_shuffle = int(tile_m) // CShuffleMLane

    c_nlane = fx.Index(CShuffleNLane)
    m_lane = tx // c_nlane
    n_lane = tx % c_nlane
    c_evec = fx.Index(EVec)

    if frag_elem_type is None:
        frag_elem_type = T.f16
    vec_frag = T.vec(EVec, frag_elem_type)
    bx_m_v = bx_m
    by_n_v = by_n

    if _do_interleave:
        _wide_evec = n_reps_shuffle * EVec
        vec_frag_wide = T.vec(_wide_evec, frag_elem_type)
        c_wide_evec = fx.Index(_wide_evec)

    _use_batched_reads = batch_cshuffle_reads or _do_interleave or (
        use_vskip and llvm is not None
    )

    if not _use_batched_reads:
        for mr in range_constexpr(m_reps_shuffle):
            row_base_m = arith.constant(mr * CShuffleMLane, index=True)
            row_local = row_base_m + m_lane
            row = bx_m_v + row_local

            row_ctx_raw = (
                precompute_row(row_local=row_local, row=row)
                if precompute_row is not None
                else None
            )

            row_ctx = row_ctx_raw
            row_pred = None
            if (
                scf is not None
                and row_ctx_raw is not None
                and isinstance(row_ctx_raw, tuple)
                and len(row_ctx_raw) == 2
            ):
                row_ctx, row_pred = row_ctx_raw

            def _do_store_row():
                row_base_lds = row_local * tile_n_idx
                for nr in range_constexpr(n_reps_shuffle):
                    col_base_nr = arith.constant(
                        nr * (CShuffleNLane * EVec), index=True
                    )
                    col_pair0 = col_base_nr + (n_lane * c_evec)
                    lds_idx_pair = row_base_lds + col_pair0
                    frag = vector.load_op(vec_frag, lds_out, [lds_idx_pair])

                    store_pair(
                        row_local=row_local,
                        row=row,
                        row_ctx=row_ctx,
                        col_pair0=col_pair0,
                        col_g0=by_n_v + col_pair0,
                        frag=frag,
                    )

            if row_pred is not None:
                _if_row = scf.IfOp(row_pred)
                with _if_then(_if_row, scf):
                    _do_store_row()
            else:
                _do_store_row()
        return

    # Phase 1: collect row metadata up-front.
    _pc_results = []
    for mr in range_constexpr(m_reps_shuffle):
        row_base_m = arith.constant(mr * CShuffleMLane, index=True)
        row_local = row_base_m + m_lane
        row = bx_m_v + row_local

        row_ctx_raw = (
            precompute_row(row_local=row_local, row=row)
            if precompute_row is not None
            else None
        )

        # Optional row-level predicate: if `precompute_row` returns `(ctx, pred_i1)` and `scf`
        # is provided, we can skip the entire N-loop for invalid rows (cheaper than per-store checks).
        row_ctx = row_ctx_raw
        row_pred = None
        if (
            scf is not None
            and row_ctx_raw is not None
            and isinstance(row_ctx_raw, tuple)
            and len(row_ctx_raw) == 2
        ):
            row_ctx, row_pred = row_ctx_raw
        _pc_results.append((row_local, row, row_ctx, row_pred))

    # Pack per-row validity into a single i32 SGPR bitmap for s_setvskip.
    _vskip_bitmap_raw = None
    if use_vskip and llvm is not None:
        _c0_vskip = arith.constant(0, index=True)
        _c1_vskip = arith.constant(1, index=True)
        bitmap_idx = _c0_vskip
        for _bmr in range_constexpr(m_reps_shuffle):
            _, _, _, _rp = _pc_results[_bmr]
            if _rp is not None:
                _not_pred = arith.select(_rp, _c0_vskip, _c1_vskip)
                _bit_weight = arith.constant(1 << _bmr, index=True)
                bitmap_idx = bitmap_idx + _not_pred * _bit_weight
        _vskip_bitmap_i32 = arith.index_cast(T.i32, bitmap_idx)
        _vskip_bitmap_raw = (
            _vskip_bitmap_i32._value
            if hasattr(_vskip_bitmap_i32, "_value")
            else _vskip_bitmap_i32
        )

    # Phase 2: issue all LDS reads back-to-back.
    _all_frags = []
    for mr in range_constexpr(m_reps_shuffle):
        row_local = _pc_results[mr][0]
        row_base_lds = row_local * tile_n_idx
        if _do_interleave:
            lds_idx_base = row_base_lds + (n_lane * c_wide_evec)
            wide_frag = vector.load_op(vec_frag_wide, lds_out, [lds_idx_base])
            _all_frags.append(wide_frag)
        else:
            row_frags = []
            for nr in range_constexpr(n_reps_shuffle):
                col_base_nr = arith.constant(nr * (CShuffleNLane * EVec), index=True)
                col_pair0 = col_base_nr + (n_lane * c_evec)
                lds_idx_pair = row_base_lds + col_pair0
                frag = vector.load_op(vec_frag, lds_out, [lds_idx_pair])
                row_frags.append(frag)
            _all_frags.append(row_frags)

    # Phase 3: address computation plus final store/atomic.
    for mr in range_constexpr(m_reps_shuffle):
        row_local, row, row_ctx, row_pred = _pc_results[mr]

        def _do_store_row():
            if _do_interleave:
                wide_frag = _all_frags[mr]
                for nr in range_constexpr(n_reps_shuffle):
                    elems = []
                    for e in range_constexpr(EVec):
                        elems.append(
                            vector.extract(
                                wide_frag,
                                static_position=[nr * EVec + e],
                                dynamic_position=[],
                            )
                        )
                    frag = vector.from_elements(vec_frag, elems)
                    col_base_nr = arith.constant(nr * (CShuffleNLane * EVec), index=True)
                    col_pair0 = col_base_nr + (n_lane * c_evec)
                    store_pair(
                        row_local=row_local,
                        row=row,
                        row_ctx=row_ctx,
                        col_pair0=col_pair0,
                        col_g0=by_n_v + col_pair0,
                        frag=frag,
                    )
            else:
                for nr in range_constexpr(n_reps_shuffle):
                    frag = _all_frags[mr][nr]
                    col_base_nr = arith.constant(nr * (CShuffleNLane * EVec), index=True)
                    col_pair0 = col_base_nr + (n_lane * c_evec)
                    store_pair(
                        row_local=row_local,
                        row=row,
                        row_ctx=row_ctx,
                        col_pair0=col_pair0,
                        col_g0=by_n_v + col_pair0,
                        frag=frag,
                    )

        if (
            row_pred is not None
            and use_vskip
            and llvm is not None
            and _vskip_bitmap_raw is not None
        ):
            pred_i32 = arith.index_cast(
                T.i32,
                arith.select(
                    row_pred,
                    arith.constant(1, index=True),
                    arith.constant(0, index=True),
                ),
            )
            pred_raw = (
                pred_i32._value if hasattr(pred_i32, "_value") else pred_i32
            )
            i64_ty = ir.IntegerType.get_signless(64)
            i32_ty = ir.IntegerType.get_signless(32)
            saved_exec_op = llvm.InlineAsmOp(
                res=i64_ty,
                operands_=[pred_raw],
                asm_string=(
                    "s_mov_b64 $0, exec\n"
                    "v_cmp_ne_u32_e64 vcc, $1, 0\n"
                    "s_and_b64 exec, exec, vcc"
                ),
                constraints="=&s,v,~{vcc}",
                has_side_effects=True,
                is_align_stack=False,
            )
            llvm.InlineAsmOp(
                res=i32_ty,
                operands_=[_vskip_bitmap_raw],
                asm_string=(
                    f"v_readfirstlane_b32 $0, $1\n"
                    f"s_setvskip $0, {mr}"
                ),
                constraints="=s,v",
                has_side_effects=True,
                is_align_stack=False,
            )
            _do_store_row()
            llvm.InlineAsmOp(
                res=None,
                operands_=[saved_exec_op.result],
                asm_string="s_setvskip 0, 0\ns_mov_b64 exec, $0",
                constraints="s",
                has_side_effects=True,
                is_align_stack=False,
            )
        elif row_pred is not None:
            _if_row = scf.IfOp(row_pred)
            with _if_then(_if_row, scf):
                _do_store_row()
        else:
            _do_store_row()


def mfma_epilog(
    *,
    use_cshuffle: bool,
    # Common (always required)
    arith,
    range_constexpr,
    m_repeat: int,
    lane_div_16,
    bx_m,
    # Default epilog (required when use_cshuffle=False)
    body_row: Callable | None = None,
    # CShuffle epilog (required when use_cshuffle=True)
    vector=None,
    gpu=None,
    scf=None,
    tile_m: int | None = None,
    tile_n: int | None = None,
    e_vec: int = 2,
    cshuffle_nlane: int = 32,
    block_size: int = 256,
    num_acc_n: int | None = None,
    tx=None,
    lane_mod_16=None,
    by_n=None,
    n_tile_base=None,
    lds_out=None,
    write_row_to_lds: Callable | None = None,
    precompute_row: Callable | None = None,  
    store_pair: Callable | None = None,
    frag_elem_type: ir.Type | None = None,
    skip_initial_barrier: bool = False,
    use_vskip: bool = False,
    llvm=None,
    batch_cshuffle_reads: bool = False,
):
    if not use_cshuffle:
        if body_row is None:
            raise ValueError("mfma_epilog(use_cshuffle=False) requires `body_row`.")
        return default_epilog(
            arith=arith,
            range_constexpr=range_constexpr,
            m_repeat=m_repeat,
            lane_div_16=lane_div_16,
            bx_m=bx_m,
            body_row=body_row,
        )

    return c_shuffle_epilog(
        arith=arith,
        vector=vector,
        gpu=gpu,
        scf=scf,
        range_constexpr=range_constexpr,
        tile_m=int(tile_m),
        tile_n=int(tile_n),
        e_vec=int(e_vec),
        cshuffle_nlane=int(cshuffle_nlane),
        block_size=int(block_size),
        m_repeat=m_repeat,
        num_acc_n=int(num_acc_n),
        tx=tx,
        lane_div_16=lane_div_16,
        lane_mod_16=lane_mod_16,
        bx_m=bx_m,
        by_n=by_n,
        n_tile_base=n_tile_base,
        lds_out=lds_out,
        frag_elem_type=frag_elem_type,
        write_row_to_lds=write_row_to_lds,
        precompute_row=precompute_row,
        store_pair=store_pair,
        skip_initial_barrier=skip_initial_barrier,
        use_vskip=use_vskip,
        llvm=llvm,
        batch_cshuffle_reads=batch_cshuffle_reads,
    )
