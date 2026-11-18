import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.moe.moe_routing.bitmatrix import (
    _sum_bitmatrix_rows_fused,
)
from aiter.ops.triton._triton_kernels.moe.moe_routing.expt_data import (
    _cdiv_pow2,
    _expt_data_compute_stage1,
    _expt_data_compute_stage2,
    _expt_data_compute_stage2_fused,
)
from aiter.ops.triton._triton_kernels.moe.moe_routing.utils import keyed_add
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit
def _routing_compute_indx(
    pid_m,
    GatherIndx,
    ScatterIndx,
    GateScal,
    ExptScal,
    ExptIndx,
    PartialOffs,
    stride_pm,
    stride_pn,
    TokensStart,
    n_gates,
    BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    N_EXPTS_ACT_PAD: tl.constexpr,
    USE_TDM: tl.constexpr,
):

    tl.static_assert(N_EXPTS_ACT_PAD * BLOCK_M <= 32768)

    LOAD_SIZE: tl.constexpr = N_EXPTS_ACT_PAD * BLOCK_M
    local_offs = tl.arange(0, LOAD_SIZE)
    offs = pid_m * BLOCK_M * N_EXPTS_ACT + local_offs
    # TDM tensor descriptors require >=16 bytes in the last dim. The expert-index
    # load is int16 (2 bytes), so LOAD_SIZE must be >=8 elements; for tiny routing
    # tiles (e.g. decode bs=1, where BLOCK_M=1 -> LOAD_SIZE=N_EXPTS_ACT_PAD) fall
    # back to the functionally-identical plain-load branch below.
    if USE_TDM and EVEN_M and N_EXPTS_ACT == N_EXPTS_ACT_PAD and LOAD_SIZE >= 8:
        expt_desc = tl.make_tensor_descriptor(
            base=ExptIndx + pid_m * BLOCK_M * N_EXPTS_ACT,
            shape=(LOAD_SIZE,),
            strides=(1,),
            block_shape=(LOAD_SIZE,),
        )
        expert = expt_desc.load([0])
        expert = tl.where(offs < n_gates, expert, -1).to(tl.uint32)
    elif EVEN_M and N_EXPTS_ACT == N_EXPTS_ACT_PAD:
        expert = tl.load(ExptIndx + offs).to(tl.uint32)
    else:
        expert = tl.load(ExptIndx + offs, mask=(offs < n_gates), other=-1).to(tl.uint32)

    # stable-sort by expert ID:
    kv_pairs = ((expert << 16) | local_offs).to(tl.uint32)
    kv_pairs = tl.sort(kv_pairs, 0)
    expert = kv_pairs >> 16
    offs = pid_m * BLOCK_M * N_EXPTS_ACT + (kv_pairs & 0xFFFF)

    if EVEN_M and N_EXPTS_ACT == N_EXPTS_ACT_PAD:
        gate_scal = tl.load(ExptScal + offs)

        # compute run lengths in expert-sorted order:
        x = kv_pairs & 0xFFFF0000 | 0x00000001
        expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, keyed_add)
        exclusive_run_lengths = (expts_and_inclusive_run_lengths - 1) & 0xFFFF

        gates = tl.load(PartialOffs + pid_m * stride_pm + expert * stride_pn)
        gates += tl.load(TokensStart + expert)
        gates += exclusive_run_lengths

        tl.store(ScatterIndx + offs, gates)
        tl.store(GatherIndx + gates, offs)
        tl.store(GateScal + gates, gate_scal)
    else:
        mask = expert != 0xFFFF
        gate_scal = tl.load(ExptScal + offs, mask=mask)

        # compute run lengths in expert-sorted order:
        x = kv_pairs & 0xFFFF0000 | 0x00000001
        expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, keyed_add)
        exclusive_run_lengths = (expts_and_inclusive_run_lengths - 1) & 0xFFFF

        gates = tl.load(PartialOffs + pid_m * stride_pm + expert * stride_pn, mask=mask)
        gates += tl.load(TokensStart + expert, mask=mask)
        gates += exclusive_run_lengths

        tl.store(ScatterIndx + offs, gates, mask=mask)
        tl.store(GatherIndx + gates, offs, mask=mask)
        tl.store(GateScal + gates, gate_scal, mask=mask)


@triton.jit
def _routing_compute_indx_fused(
    GatherIndx,
    ScatterIndx,
    GateScal,
    ExptScal,
    ExptIndx,
    TokensStart,
    n_gates,
    BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    N_EXPTS_ACT_PAD: tl.constexpr,
    USE_TDM: tl.constexpr,
):

    tl.static_assert(N_EXPTS_ACT_PAD * BLOCK_M <= 32768)

    LOAD_SIZE: tl.constexpr = N_EXPTS_ACT_PAD * BLOCK_M
    local_offs = tl.arange(0, LOAD_SIZE)
    offs = local_offs
    # TDM tensor descriptors require >=16 bytes in the last dim. The expert-index
    # load is int16 (2 bytes), so LOAD_SIZE must be >=8 elements; for tiny routing
    # tiles (e.g. decode bs=1, where BLOCK_M=1 -> LOAD_SIZE=N_EXPTS_ACT_PAD) fall
    # back to the functionally-identical plain-load branch below.
    if USE_TDM and EVEN_M and N_EXPTS_ACT == N_EXPTS_ACT_PAD and LOAD_SIZE >= 8:
        expt_desc = tl.make_tensor_descriptor(
            base=ExptIndx,
            shape=(LOAD_SIZE,),
            strides=(1,),
            block_shape=(LOAD_SIZE,),
        )
        expert = expt_desc.load([0])
        expert = tl.where(offs < n_gates, expert, -1).to(tl.uint32)
    elif EVEN_M and N_EXPTS_ACT == N_EXPTS_ACT_PAD:
        expert = tl.load(ExptIndx + offs).to(tl.uint32)
    else:
        expert = tl.load(ExptIndx + offs, mask=(offs < n_gates), other=-1).to(tl.uint32)

    # stable-sort by expert ID:
    kv_pairs = ((expert << 16) | local_offs).to(tl.uint32)
    kv_pairs = tl.sort(kv_pairs, 0)
    expert = kv_pairs >> 16
    offs = kv_pairs & 0xFFFF

    if EVEN_M and N_EXPTS_ACT == N_EXPTS_ACT_PAD:
        gate_scal = tl.load(ExptScal + offs)

        # compute run lengths in expert-sorted order:
        x = kv_pairs & 0xFFFF0000 | 0x00000001
        expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, keyed_add)
        exclusive_run_lengths = (expts_and_inclusive_run_lengths - 1) & 0xFFFF

        gates = tl.load(TokensStart + expert)
        gates += exclusive_run_lengths

        tl.store(ScatterIndx + offs, gates)
        tl.store(GatherIndx + gates, offs)
        tl.store(GateScal + gates, gate_scal)
    else:
        mask = expert != 0xFFFF
        gate_scal = tl.load(ExptScal + offs, mask=mask)

        # compute run lengths in expert-sorted order:
        x = kv_pairs & 0xFFFF0000 | 0x00000001
        expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, keyed_add)
        exclusive_run_lengths = (expts_and_inclusive_run_lengths - 1) & 0xFFFF

        gates = tl.load(TokensStart + expert, mask=mask)
        gates += exclusive_run_lengths

        tl.store(ScatterIndx + offs, gates, mask=mask)
        tl.store(GatherIndx + gates, offs, mask=mask)
        tl.store(GateScal + gates, gate_scal, mask=mask)


_combined_routing_repr = make_kernel_repr(
    "_combined_routing",
    [
        "BLOCK_M",
        "EVEN_M",
        "N_EXPTS_ACT",
        "N_EXPTS_ACT_PAD",
        "n_expts_tot",
        "tile_dim_log2",
        "BLOCK_A",
        "EQUAL_A",
        "USE_TDM",
    ],
)


@triton.jit(repr=_combined_routing_repr)
def _combined_routing(
    GatherIndx,
    ScatterIndx,
    GateScal,
    ExptScal,
    ExptIndx,
    PartialOffs,
    stride_pm,
    stride_pn,
    n_gates,
    BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    N_EXPTS_ACT_PAD: tl.constexpr,
    ExpertHist,
    n_expts_tot: tl.constexpr,
    TokenStart,
    TileStart,
    blocks1a,
    MDTileInfo,
    max_num_tiles,
    tile_dim_log2: tl.constexpr,
    BLOCK_A: tl.constexpr,
    EQUAL_A: tl.constexpr,
    USE_TDM: tl.constexpr,
):

    pid = tl.program_id(0)

    if pid != 0 and pid < blocks1a:  # noqa: SIM102
        if tl.load(ExpertHist + pid) == 0:
            return

    tile_start = _expt_data_compute_stage1(
        pid,
        ExpertHist,
        n_expts_tot,
        TokenStart,
        TileStart,
        MDTileInfo,
        max_num_tiles,
        n_gates,
        tile_dim_log2,
        BLOCK_A,
        EQUAL_A,
    )

    if pid < blocks1a:
        _expt_data_compute_stage2(
            pid, ExpertHist, tile_start, MDTileInfo, tile_dim_log2
        )
    else:
        pid -= blocks1a
        _routing_compute_indx(
            pid,
            GatherIndx,
            ScatterIndx,
            GateScal,
            ExptScal,
            ExptIndx,
            PartialOffs,
            stride_pm,
            stride_pn,
            TokenStart,
            n_gates,
            BLOCK_M,
            EVEN_M,
            N_EXPTS_ACT,
            N_EXPTS_ACT_PAD,
            USE_TDM,
        )


_combined_routing_fused_repr = make_kernel_repr(
    "_combined_routing_fused",
    [
        "BLOCK_M",
        "EVEN_M",
        "N_EXPTS_ACT",
        "N_EXPTS_ACT_PAD",
        "N_EXPTS_TOT",
        "N_BLKS_BITMATRIX",
        "tile_dim_log2",
        "BLOCK_A",
        "EQUAL_A",
        "USE_TDM",
    ],
)


@triton.jit(repr=_combined_routing_fused_repr)
def _combined_routing_fused(
    GatherIndx,
    ScatterIndx,
    GateScal,
    ExptScal,
    ExptIndx,
    Bitmatrix,
    shape_bm,
    stride_bm,
    stride_bn,
    N_BLKS_BITMATRIX: tl.constexpr,
    n_gates,
    BLOCK_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    N_EXPTS_ACT_PAD: tl.constexpr,
    N_EXPTS_TOT: tl.constexpr,
    ExpertHist,
    TokenStart,
    TileStart,
    blocks1a,
    MDTileInfo,
    max_num_tiles,
    tile_dim_log2: tl.constexpr,
    BLOCK_A: tl.constexpr,
    EQUAL_A: tl.constexpr,
    USE_TDM: tl.constexpr,
):

    pid = tl.program_id(0)

    _sum_bitmatrix_rows_fused(
        Bitmatrix,
        shape_bm,
        stride_bm,
        stride_bn,
        ExpertHist,
        N_BLKS_BITMATRIX,
        BLOCK_M,
        EVEN_M,
    )

    tl.debug_barrier()

    if pid != 0 and pid < blocks1a:
        n_tokens = tl.load(ExpertHist + pid)
        if n_tokens == 0:
            return

    tile_start = _expt_data_compute_stage1(
        pid,
        ExpertHist,
        N_EXPTS_TOT,
        TokenStart,
        TileStart,
        MDTileInfo,
        max_num_tiles,
        n_gates,
        tile_dim_log2,
        BLOCK_A,
        EQUAL_A,
    )

    if pid < blocks1a:
        _expt_data_compute_stage2_fused(pid, ExpertHist, tile_start, MDTileInfo)
    else:
        _routing_compute_indx_fused(
            GatherIndx,
            ScatterIndx,
            GateScal,
            ExptScal,
            ExptIndx,
            TokenStart,
            n_gates,
            BLOCK_M,
            EVEN_M,
            N_EXPTS_ACT,
            N_EXPTS_ACT_PAD,
            USE_TDM,
        )


# -----------------------------------------------------------------------------
# expert-parallel sort: routing over ALREADY-DISPATCHED rows
#
# The kernels below serve `ep_sort_routing`, which starts from the output of
# an expert-parallel all-to-all instead of from router logits. The top-k choice
# is already made and the rows have been permuted across ranks, so there is no
# bitmatrix to reduce and only *some* gates of a given row belong to this rank.
#
# A CTA owns BLOCK consecutive gates. It counts them per expert (one histogram
# per CTA, no per-gate atomics) and ranks them by a sort of (expert, gate) keys.
# Each expert's first row comes from the per-CTA counts. So each expert's rows
# are in ascending gate order, and the result does not depend on the CTA
# schedule.
# -----------------------------------------------------------------------------


@triton.jit
def _ep_gate_limit(NumLocalTokens, n_gates, TOPK: tl.constexpr):
    # Gates of rows past NumLocalTokens are dead (receive-buffer padding).
    lim = n_gates
    if NumLocalTokens is not None:
        lim = tl.minimum(lim, tl.load(NumLocalTokens) * TOPK)
    return lim


@triton.jit
def _ep_gate_experts(
    DispatchIds, ExpertMap, offs, lim, e_map_numel, SENTINEL: tl.constexpr
):
    in_lim = offs < lim
    ids = tl.load(DispatchIds + offs, mask=in_lim, other=0)
    ids = tl.minimum(tl.maximum(ids, 0), e_map_numel - 1)
    local = tl.load(ExpertMap + ids, mask=in_lim, other=-1)
    valid = (local >= 0) & in_lim
    return valid, tl.where(valid, local, SENTINEL)


@triton.jit
def _ep_hist(expt, valid, N_BINS: tl.constexpr, SENTINEL: tl.constexpr):
    # Counts of the valid gates. With N_BINS == SENTINEL the dead gates have no bin.
    if SENTINEL >= N_BINS:
        expt = tl.where(valid, expt, 0)
    return tl.histogram(expt, N_BINS, mask=valid)


@triton.jit
def _ep_tile_offsets(
    h,
    bins,
    TokenStart,
    TileStart,
    do_store,
    N_EXPTS: tl.constexpr,
    tile_dim_log2: tl.constexpr,
):
    # ExptData stage1 from the histogram in registers: first row and first tile
    # of each expert.
    is_expt = bins < N_EXPTS
    tl.store(TokenStart + bins, tl.cumsum(h, 0) - h, mask=is_expt & do_store)
    tl.store(TokenStart + N_EXPTS, tl.sum(h, 0), mask=do_store)
    n_tiles = tl.where(is_expt, _cdiv_pow2(h, tile_dim_log2), 0)
    tiles_incl = tl.cumsum(n_tiles, 0)
    tl.store(TileStart + bins, tiles_incl - n_tiles, mask=is_expt & do_store)
    total_tiles = tl.sum(n_tiles, 0)
    tl.store(TileStart + N_EXPTS, total_tiles, mask=do_store)
    return n_tiles, tiles_incl, total_tiles


@triton.jit
def _ep_scatter_sorted(
    expt,
    start,
    expt_base,
    n_live,
    n_gates,
    DispatchWeights,
    GatherIndx,
    ScatterIndx,
    GateScal,
    DstRow,
    SrcTokenMap,
    N_EXPTS: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG2_BLOCK: tl.constexpr,
    HAS_DST_ROW: tl.constexpr,
    FILL_DST_TAIL: tl.constexpr,
    TOPK: tl.constexpr,
    MAX_TOK: tl.constexpr,
    PEER_ROWS: tl.constexpr,
):
    # Sort the CTA's gates by (expert, gate). Sorted index i of expert e goes to
    # row expt_base[e] + i. Dead gates (SENTINEL) sort to the end.
    loc = tl.arange(0, BLOCK)
    keys = tl.sort((expt << LOG2_BLOCK) | loc)
    s_expt = keys >> LOG2_BLOCK
    s_gate = start + (keys & (BLOCK - 1))
    s_live = s_expt < N_EXPTS
    # Dead gates (s_expt == SENTINEL) read any bin: they store nothing.
    pos = tl.gather(expt_base, tl.minimum(s_expt, N_EXPTS - 1), 0) + loc
    tl.store(GatherIndx + pos, s_gate, mask=s_live)
    w = tl.load(DispatchWeights + s_gate, mask=s_live, other=0.0)
    tl.store(GateScal + pos, w.to(tl.float32), mask=s_live)
    # Dead gates get 0: the grouped reduce skips them through GateValid.
    tl.store(ScatterIndx + s_gate, tl.where(s_live, pos, 0), mask=s_gate < n_gates)
    if HAS_DST_ROW:
        # Row of (origin rank, its token, k) in the combine staging window.
        recv_slot = s_gate // TOPK
        k = s_gate - recv_slot * TOPK
        enc = tl.load(SrcTokenMap + recv_slot, mask=s_live, other=0)
        origin_pe = enc // MAX_TOK
        dst = origin_pe * PEER_ROWS + (enc - origin_pe * MAX_TOK) * TOPK + k
        tl.store(DstRow + pos, dst, mask=s_live)
        if FILL_DST_TAIL:
            # Rows past the live ones must not be delivered. This CTA fills the
            # ones in its own gate range; no other CTA writes them.
            row = start + loc
            tl.store(DstRow + row, -1, mask=(row >= n_live) & (row < n_gates))


_ep_sort_fused_repr = make_kernel_repr(
    "_ep_sort_fused_kernel",
    ["N_EXPTS", "TOPK", "N_BINS", "BLOCK", "HAS_DST_ROW"],
)


@triton.jit(repr=_ep_sort_fused_repr)
def _ep_sort_fused_kernel(
    DispatchIds,  # (M, topk) int32, GLOBAL expert ids
    ExpertMap,  # (E_map,) int32
    NumLocalTokens,  # (1,) int32, or None when every row is live
    DispatchWeights,  # (M, topk) f32, read flat
    SrcTokenMap,  # (max_recv,) int32, recv slot -> origin_pe*MAX_TOK + lid
    GateValid,  # (G,) int32 out
    Hist,  # (N_BINS,) int32 out
    TokenStart,  # (N_EXPTS+1,) int32 out == token_offs_raw
    TileStart,  # (N_EXPTS+1,) int32 out == token_offs_pad
    MDTileInfo,  # (max_num_tiles,) int32 out == block_pid_map
    GatherIndx,  # (G,) int32 out
    ScatterIndx,  # (G,) int32 out
    GateScal,  # (G,) f32 out
    DstRow,  # (G,) int32 out, EP scatter destinations
    max_num_tiles,
    n_gates,
    e_map_numel,
    N_EXPTS: tl.constexpr,
    TOPK: tl.constexpr,
    SENTINEL: tl.constexpr,
    N_BINS: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG2_BLOCK: tl.constexpr,
    tile_dim_log2: tl.constexpr,
    TILE_BLOCK: tl.constexpr,
    MEMSET_BLOCK: tl.constexpr,
    HAS_DST_ROW: tl.constexpr,
    MAX_TOK: tl.constexpr,
    PEER_ROWS: tl.constexpr,
):
    """EP sort in one launch of one CTA, for small n_gates (decode). The CTA has
    all gates, so it needs no scratch, no atomics and no join."""
    lim = _ep_gate_limit(NumLocalTokens, n_gates, TOPK)
    bins = tl.arange(0, N_BINS)
    offs = tl.arange(0, BLOCK)
    valid, expt = _ep_gate_experts(
        DispatchIds, ExpertMap, offs, lim, e_map_numel, SENTINEL
    )
    h = _ep_hist(expt, valid, N_BINS, SENTINEL)
    tl.store(Hist + bins, h)
    n_tiles, tiles_incl, total_tiles = _ep_tile_offsets(
        h, bins, TokenStart, TileStart, True, N_EXPTS, tile_dim_log2
    )
    # MDTileInfo: (tile << 16) + expert, then -1 up to max_num_tiles.
    tile_first = tiles_incl - n_tiles
    for j0 in range(0, tl.max(n_tiles, 0), TILE_BLOCK):
        j = j0 + tl.arange(0, TILE_BLOCK)
        tl.store(
            MDTileInfo + tile_first[:, None] + j[None, :],
            (j[None, :] << 16) + bins[:, None],
            mask=j[None, :] < n_tiles[:, None],
        )
    for t0 in range(total_tiles, max_num_tiles, MEMSET_BLOCK):
        t = t0 + tl.arange(0, MEMSET_BLOCK)
        tl.store(MDTileInfo + t, -1, mask=t < max_num_tiles)

    tl.store(GateValid + offs, valid.to(tl.int32), mask=offs < n_gates)
    # One CTA: the sorted index is the row.
    _ep_scatter_sorted(
        expt,
        0,
        tl.zeros([N_BINS], dtype=tl.int32),
        tl.sum(h, 0),
        n_gates,
        DispatchWeights,
        GatherIndx,
        ScatterIndx,
        GateScal,
        DstRow,
        SrcTokenMap,
        N_EXPTS,
        BLOCK,
        LOG2_BLOCK,
        HAS_DST_ROW,
        True,
        TOPK,
        MAX_TOK,
        PEER_ROWS,
    )


_ep_gate_prep_scan_repr = make_kernel_repr(
    "_ep_gate_prep_scan_kernel",
    ["N_EXPTS", "TOPK", "N_BINS", "BLOCK", "HAS_DST_ROW", "JOIN"],
)


@triton.jit(repr=_ep_gate_prep_scan_repr)
def _ep_gate_prep_scan_kernel(
    DispatchIds,  # (M, topk) int32, GLOBAL expert ids
    ExpertMap,  # (E_map,) int32
    NumLocalTokens,  # (1,) int32, or None to skip the row mask entirely.
    GateValid,  # (G,) int32 out
    CtaHist,  # (n_ctas, N_BINS) int32 out, per-CTA histogram
    HistAtomic,  # (N_BINS,) int32 scratch, ZERO on entry, left ZERO on exit
    Ticket,  # (1,) int32 scratch, ditto -- the grid-wide join counter
    Hist,  # (N_BINS,) int32 out, the durable histogram (JOIN only)
    Cursor,  # (N_BINS,) int32 out, first row of each expert (JOIN only)
    TokenStart,  # (N_EXPTS+1,) int32 out == token_offs_raw (JOIN only)
    TileStart,  # (N_EXPTS+1,) int32 out == token_offs_pad (JOIN only)
    MDTileInfo,  # (max_num_tiles,) int32 out == block_pid_map, tail (JOIN only)
    DstRow,  # (G,) int32 out, pre-filled with -1 here (JOIN only)
    max_num_tiles,
    n_gates,
    e_map_numel,
    N_EXPTS: tl.constexpr,
    TOPK: tl.constexpr,
    SENTINEL: tl.constexpr,
    N_BINS: tl.constexpr,
    BLOCK: tl.constexpr,
    tile_dim_log2: tl.constexpr,
    MEMSET_BLOCK: tl.constexpr,
    HAS_DST_ROW: tl.constexpr,
    JOIN: tl.constexpr,
):
    """EP sort kernel A: gating + per-CTA histogram (CtaHist).

    Without JOIN that is all: each CTA of kernel B sums CtaHist itself.

    With JOIN (very large grids) each CTA also adds its histogram to HistAtomic
    with one atomic per non-empty expert, then takes a ticket. The CTA that draws
    the last ticket owns the scan and ExptData stage1. No CTA waits for another,
    so any occupancy is safe. HistAtomic and Ticket are left zeroed for the next
    call.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_gates
    if HAS_DST_ROW and JOIN:
        # Kernel B overwrites the live rows; the tail must stay negative.
        tl.store(DstRow + offs, -1, mask=mask)
    lim = _ep_gate_limit(NumLocalTokens, n_gates, TOPK)
    bins = tl.arange(0, N_BINS)
    if pid * BLOCK < lim:
        valid, expt = _ep_gate_experts(
            DispatchIds, ExpertMap, offs, lim, e_map_numel, SENTINEL
        )
        tl.store(GateValid + offs, valid.to(tl.int32), mask=mask)
        h = _ep_hist(expt, valid, N_BINS, SENTINEL)
        tl.store(CtaHist + pid * N_BINS + bins, h)
        if JOIN:
            # Relaxed: the barrier below waits for these atomics, and the ticket
            # (acq_rel) is taken after it.
            tl.atomic_add(HistAtomic + bins, h, mask=h > 0, sem="relaxed", scope="gpu")
    else:
        # Every gate of this CTA is past NumLocalTokens.
        tl.store(GateValid + offs, 0, mask=mask)
        if not JOIN:
            tl.store(CtaHist + pid * N_BINS + bins, 0)
    if JOIN:
        # Finish all waves' histogram atomics before this CTA takes its ticket.
        tl.debug_barrier()
        # Grid-wide join. The last CTA sees every CTA's histogram atomics.
        one = tl.arange(0, 1)
        prev = tl.atomic_add(Ticket + one, 1, sem="acq_rel", scope="gpu")
        if tl.sum(prev, 0) == tl.num_programs(0) - 1:
            h = tl.load(HistAtomic + bins)
            tl.store(Hist + bins, h)
            tl.store(Cursor + bins, tl.cumsum(h, 0) - h)
            # Re-arm the scratch for the next call. Safe here and only here:
            # drawing the last ticket means every other CTA is done with both.
            tl.store(HistAtomic + bins, 0)
            tl.store(Ticket + one, 0)
            _, _, total_tiles = _ep_tile_offsets(
                h, bins, TokenStart, TileStart, True, N_EXPTS, tile_dim_log2
            )
            for t0 in range(total_tiles, max_num_tiles, MEMSET_BLOCK):
                t = t0 + tl.arange(0, MEMSET_BLOCK)
                tl.store(MDTileInfo + t, -1, mask=t < max_num_tiles)


_ep_scatter_expt_data_repr = make_kernel_repr(
    "_ep_scatter_expt_data_kernel",
    ["N_EXPTS", "TOPK", "N_BINS", "BLOCK", "HAS_DST_ROW", "JOIN"],
)


@triton.jit(repr=_ep_scatter_expt_data_repr)
def _ep_scatter_expt_data_kernel(
    DispatchIds,  # (M, topk) int32, GLOBAL expert ids
    ExpertMap,  # (E_map,) int32
    NumLocalTokens,  # (1,) int32, or None when every row is live
    DispatchWeights,  # (M, topk) f32 in, read flat
    CtaHist,  # (n_ctas, N_BINS) int32 in
    Cursor,  # (N_BINS,) int32 in, first row of each expert (JOIN only)
    GatherIndx,  # (G,) int32 out
    ScatterIndx,  # (G,) int32 out
    GateScal,  # (G,) f32 out
    Hist,  # (N_BINS,) int32 in (JOIN) or out
    TokenStart,  # (N_EXPTS+1,) int32 out == token_offs_raw (without JOIN)
    TileStart,  # (N_EXPTS+1,) int32 in (JOIN) or out == token_offs_pad
    MDTileInfo,  # (max_num_tiles,) int32 out == block_pid_map
    # --- EP scatter map, scatter half only (see ep_sort_routing) ---
    DstRow,  # (G,) int32 out, indexed by sorted position
    SrcTokenMap,  # (max_recv,) int32 in, recv slot -> origin_pe*MAX_TOK + lid
    max_num_tiles,
    n_gates,
    e_map_numel,
    n_ctas,
    N_EXPTS: tl.constexpr,
    TOPK: tl.constexpr,
    SENTINEL: tl.constexpr,
    N_BINS: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG2_BLOCK: tl.constexpr,
    PREFIX_ROWS: tl.constexpr,
    tile_dim_log2: tl.constexpr,
    MEMSET_BLOCK: tl.constexpr,
    HAS_DST_ROW: tl.constexpr,
    JOIN: tl.constexpr,
    MAX_TOK: tl.constexpr,
    PEER_ROWS: tl.constexpr,
):
    """EP sort kernel B: sorted scatter | ExptData stage2, split by CTA index.

    The first N_EXPTS CTAs write MDTileInfo. A scatter CTA ranks its gates by a
    sort; each expert's first row is its offset plus the counts of the CTAs in
    front (rows in ascending gate order). Without JOIN every CTA sums CtaHist
    itself, and also writes the ExptData stage1 outputs of kernel A.
    """
    pid = tl.program_id(0)
    bins = tl.arange(0, N_BINS)
    if not JOIN:
        # Every CTA sums the whole CtaHist (n_ctas x N_EXPTS) itself; a scatter
        # CTA also sums the rows of the CTAs in front of it.
        cta = pid - N_EXPTS
        total = tl.zeros([N_BINS], dtype=tl.int32)
        ahead = tl.zeros([N_BINS], dtype=tl.int32)
        for r0 in range(0, n_ctas, PREFIX_ROWS):
            rows = r0 + tl.arange(0, PREFIX_ROWS)
            part = tl.load(
                CtaHist + rows[:, None] * N_BINS + bins[None, :],
                mask=(rows[:, None] < n_ctas) & (bins[None, :] < N_EXPTS),
                other=0,
            )
            total += tl.sum(part, 0)
            ahead += tl.sum(tl.where(rows[:, None] < cta, part, 0), 0)
        n_tiles, tiles_incl, total_tiles = _ep_tile_offsets(
            total, bins, TokenStart, TileStart, pid == N_EXPTS, N_EXPTS, tile_dim_log2
        )
    if pid >= N_EXPTS:
        cta = pid - N_EXPTS
        start = cta * BLOCK
        offs = start + tl.arange(0, BLOCK)
        lim = _ep_gate_limit(NumLocalTokens, n_gates, TOPK)
        if not JOIN:
            n_live = tl.sum(total, 0)
            if cta == 0:
                tl.store(Hist + bins, total)
                for t0 in range(total_tiles, max_num_tiles, MEMSET_BLOCK):
                    t = t0 + tl.arange(0, MEMSET_BLOCK)
                    tl.store(MDTileInfo + t, -1, mask=t < max_num_tiles)
            if HAS_DST_ROW:
                tl.store(DstRow + offs, -1, mask=(offs >= n_live) & (offs < n_gates))
        if start >= lim:
            tl.store(ScatterIndx + offs, 0, mask=offs < n_gates)
            return
        _, expt = _ep_gate_experts(
            DispatchIds, ExpertMap, offs, lim, e_map_numel, SENTINEL
        )
        h = tl.load(CtaHist + cta * N_BINS + bins)
        if not JOIN:
            first = (tl.cumsum(total, 0) - total) + ahead
        else:
            ahead = tl.zeros([N_BINS], dtype=tl.int32)
            for r0 in range(0, cta, PREFIX_ROWS):
                rows_p = r0 + tl.arange(0, PREFIX_ROWS)
                part = tl.load(
                    CtaHist + rows_p[:, None] * N_BINS + bins[None, :],
                    mask=(rows_p[:, None] < cta) & (bins[None, :] < N_EXPTS),
                    other=0,
                )
                ahead += tl.sum(part, 0)
            first = tl.load(Cursor + bins) + ahead
        _ep_scatter_sorted(
            expt,
            start,
            first - (tl.cumsum(h, 0) - h),
            0,
            n_gates,
            DispatchWeights,
            GatherIndx,
            ScatterIndx,
            GateScal,
            DstRow,
            SrcTokenMap,
            N_EXPTS,
            BLOCK,
            LOG2_BLOCK,
            HAS_DST_ROW,
            False,
            TOPK,
            MAX_TOK,
            PEER_ROWS,
        )
    elif JOIN:
        tile_start = tl.load(TileStart + pid)
        # Last statement in the branch on purpose: stage2 early-returns for empty
        # experts, so nothing may follow it.
        _expt_data_compute_stage2(pid, Hist, tile_start, MDTileInfo, tile_dim_log2)
    else:
        # Tiles of expert `pid`.
        is_e = bins == pid
        n_t = tl.sum(tl.where(is_e, n_tiles, 0), 0)
        first_t = tl.sum(tl.where(is_e, tiles_incl - n_tiles, 0), 0)
        for j0 in range(0, n_t, 8):
            j = j0 + tl.arange(0, 8)
            tl.store(MDTileInfo + first_t + j, (j << 16) + pid, mask=j < n_t)
