import triton
import triton.language as tl
from triton.language.extra import libdevice

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
from aiter.ops.triton._triton_kernels.moe.quant_moe import _compute_mx_quant_and_scale
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_pack_bits
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
# The kernels below serve `ep_sort_routing`, which starts from the output of an
# expert-parallel all-to-all instead of from router logits. The top-k choice is
# already made and the rows have been permuted across ranks, so only *some*
# gates of a given row belong to this rank.
#
# A CTA owns BLOCK consecutive gates. It counts them per expert (one histogram
# per CTA, no per-gate atomics) and ranks them by a sort of (expert, gate) keys.
# Each CTA derives the experts' first rows from the per-CTA counts itself, like
# the FlyDSL psum remap. So each expert's rows are in ascending gate order, and
# the result does not depend on the CTA schedule.
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
def _ep_topk_rows(
    Logits,
    stride_logits: tl.constexpr,
    DispatchIds,
    DispatchWeights,
    row0,
    lim,
    n_gates,
    ROWS: tl.constexpr,
    TOPK: tl.constexpr,
    N_EXPTS: tl.constexpr,
):
    # Renormalized top-k softmax of rows row0 .. row0 + ROWS with the bits of
    # topk_softmax (OCML expf, the top-k sum in order, a reciprocal multiply).
    # Writes their ids (-1 past lim) and weights (0 past lim).
    rows = row0 + tl.arange(0, ROWS)
    in_m = rows < n_gates // TOPK
    e = tl.arange(0, N_EXPTS)
    # A row may start off a 16-B boundary (stride 257: the [:, :256] view of the
    # router + shared-gate logits). Load the aligned window that starts `lead`
    # elements early and put its next VEC elements over the first ones: slot j
    # holds expert (j - lead) % N_EXPTS, and every load stays VEC wide. The
    # stride is constexpr: for whole 16-B rows lead folds to 0 (a plain row load).
    VEC: tl.constexpr = min(8, N_EXPTS)
    row_off = rows.to(tl.int64) * stride_logits
    lead = row_off & (VEC - 1)
    win = Logits + tl.multiple_of(row_off - lead, VEC)[:, None] + e[None, :]
    v = tl.load(win, mask=in_m[:, None], other=0.0)
    wrap = tl.load(
        win + N_EXPTS,
        mask=(in_m & (lead > 0))[:, None] & (e[None, :] < VEC),
        other=0.0,
    )
    head = e[None, :] < lead[:, None]
    v = tl.where(head, wrap, v).to(tl.float32)
    expt = (e[None, :] - lead[:, None].to(tl.int32)) & (N_EXPTS - 1)
    vmax = tl.max(v, 1)
    k = tl.arange(0, TOPK)
    ids = tl.zeros([ROWS, TOPK], dtype=tl.int32)
    w = tl.zeros([ROWS, TOPK], dtype=tl.float32)
    renorm = tl.zeros([ROWS], dtype=tl.float32)
    for i in tl.static_range(TOPK):
        vi = tl.max(v, 1)
        ex = tl.min(tl.where(v == vi[:, None], expt, 2147483647), 1)
        # All-NaN rows have no winner: expert 0, as topk_softmax.
        ex = tl.where(ex == 2147483647, 0, ex)
        numer = libdevice.exp(vi - vmax)
        renorm = renorm + numer
        ids = tl.where(k[None, :] == i, ex[:, None], ids)
        w = tl.where(k[None, :] == i, numer[:, None], w)
        v = tl.where(expt == ex[:, None], -float("inf"), v)
    w = w * tl.div_rn(1.0, renorm)[:, None]
    live = (rows < lim // TOPK)[:, None] & (k[None, :] < TOPK)
    gates = rows[:, None] * TOPK + k[None, :]
    tl.store(DispatchIds + gates, tl.where(live, ids, -1), mask=in_m[:, None])
    tl.store(DispatchWeights + gates, tl.where(live, w, 0.0), mask=in_m[:, None])


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
    # ExptData stage1 from the histogram in registers.
    is_expt = bins < N_EXPTS
    tl.store(TokenStart + bins, tl.cumsum(h, 0) - h, mask=is_expt & do_store)
    tl.store(TokenStart + N_EXPTS, tl.sum(h, 0), mask=do_store)
    return _ep_tile_starts(h, bins, TileStart, do_store, N_EXPTS, tile_dim_log2)


@triton.jit
def _ep_tile_starts(
    h, bins, TileStart, do_store, N_EXPTS: tl.constexpr, tile_dim_log2: tl.constexpr
):
    # First tile of each expert, for tiles of 2**tile_dim_log2 rows.
    is_expt = bins < N_EXPTS
    n_tiles = tl.where(is_expt, _cdiv_pow2(h, tile_dim_log2), 0)
    tiles_incl = tl.cumsum(n_tiles, 0)
    tl.store(TileStart + bins, tiles_incl - n_tiles, mask=is_expt & do_store)
    total_tiles = tl.sum(n_tiles, 0)
    tl.store(TileStart + N_EXPTS, total_tiles, mask=do_store)
    return n_tiles, tiles_incl, total_tiles


@triton.jit
def _ep_tile_map(
    n_tiles,
    tiles_incl,
    total_tiles,
    bins,
    mine,
    MDTileInfo,
    max_num_tiles,
    pid,
    n_ctas,
    TILE_BLOCK: tl.constexpr,
    MEMSET_BLOCK: tl.constexpr,
):
    # MDTileInfo of the experts in `mine` ((tile << 16) + expert), and this
    # CTA's share of the -1 tail.
    tile_first = tiles_incl - n_tiles
    max_tiles = tl.max(tl.where(mine, n_tiles, 0), 0)
    for j0 in range(0, max_tiles, TILE_BLOCK):
        j = j0 + tl.arange(0, TILE_BLOCK)
        tl.store(
            MDTileInfo + tile_first[:, None] + j[None, :],
            (j[None, :] << 16) + bins[:, None],
            mask=(j[None, :] < n_tiles[:, None]) & mine[:, None],
        )
    for t0 in range(
        total_tiles + pid * MEMSET_BLOCK, max_num_tiles, n_ctas * MEMSET_BLOCK
    ):
        t = t0 + tl.arange(0, MEMSET_BLOCK)
        tl.store(MDTileInfo + t, -1, mask=t < max_num_tiles)


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


@triton.jit
def _ep_mxfp4_quant_tile(
    qid,
    X,  # (M, K) bf16, unit column stride
    XQ,  # (M, K // 2) uint8 out, packed e2m1
    XS,  # (M, K // 32) uint8 out, e8m0
    NumLocalTokens,
    M,
    stride_x,
    stride_q,
    stride_s,
    K: tl.constexpr,
    QBM: tl.constexpr,
    QBN: tl.constexpr,
    LIVE_ONLY: tl.constexpr,
):
    """MXFP4 quant of one QBM x QBN tile of X. Same bytes as the gfx1250 Gluon
    dynamic_mxfp4_quant (hardware scaled cvt): e8m0 = floor(log2(rounded amax)) - 2,
    payload = RNE(x / 2^(e - 127)) to e2m1. LIVE_ONLY: rows past NumLocalTokens are
    not written."""
    NQ: tl.constexpr = QBN // 32
    N_COL: tl.constexpr = K // QBN
    pid_m = qid // N_COL
    pid_n = qid - pid_m * N_COL
    m_lim = M
    if LIVE_ONLY and NumLocalTokens is not None:
        m_lim = tl.minimum(m_lim, tl.load(NumLocalTokens))
    if pid_m * QBM < m_lim:
        rows = pid_m * QBM + tl.arange(0, QBM)
        row_ok = rows < m_lim
        rows64 = rows.to(tl.int64)
        cols = pid_n * QBN + tl.arange(0, QBN)
        x = tl.load(
            X + rows64[:, None] * stride_x + cols[None, :],
            mask=row_ok[:, None],
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        xg = x.reshape(QBM, NQ, 32)
        amax = tl.max(tl.abs(xg), axis=-1, keep_dims=True)
        a = amax.to(tl.int32, bitcast=True)
        a = (a + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
        unb = tl.log2(a.to(tl.float32, bitcast=True)).floor() - 2
        unb = tl.clamp(unb, min=-127, max=127)
        # 2^-127 is subnormal (exp2 flushes it): split it in two exact steps.
        top = unb == 127
        qx = xg * tl.exp2(-(unb - top.to(tl.float32)))
        qx = qx * tl.where(top, 0.5, 1.0)
        xq = _mxfp4_pack_bits(qx.reshape(QBM, QBN), QBN, QBM, 32)
        e8 = (unb.to(tl.int32) + 127).to(tl.uint8).reshape(QBM, NQ)
        qcols = pid_n * (QBN // 2) + tl.arange(0, QBN // 2)
        tl.store(
            XQ + rows64[:, None] * stride_q + qcols[None, :], xq, mask=row_ok[:, None]
        )
        scols = pid_n * NQ + tl.arange(0, NQ)
        tl.store(
            XS + rows64[:, None] * stride_s + scols[None, :], e8, mask=row_ok[:, None]
        )


@triton.jit
def _ep_mxfp8_quant_tile(
    qid,
    X,  # (M, K) bf16, unit column stride
    XQ,  # (M, K) float8e4nv out
    XS,  # (M, K // 32) uint8 out, e8m0
    NumLocalTokens,
    M,
    stride_x,
    stride_q,
    stride_s,
    K: tl.constexpr,
    QBM: tl.constexpr,
    QBN: tl.constexpr,
    LIVE_ONLY: tl.constexpr,
):
    """MXFP8 (e4m3) quant of one QBM x QBN tile of X with the math of
    downcast_to_mxfp(X, float8_e4m3fn) (its _compute_mx_quant_and_scale, ROUND_UP
    scale), so the bytes are the same. LIVE_ONLY: rows past NumLocalTokens are not
    written."""
    NQ: tl.constexpr = QBN // 32
    N_COL: tl.constexpr = K // QBN
    pid_m = qid // N_COL
    pid_n = qid - pid_m * N_COL
    m_lim = M
    if LIVE_ONLY and NumLocalTokens is not None:
        m_lim = tl.minimum(m_lim, tl.load(NumLocalTokens))
    if pid_m * QBM < m_lim:
        rows = pid_m * QBM + tl.arange(0, QBM)
        rows64 = rows.to(tl.int64)
        cols = pid_n * QBN + tl.arange(0, QBN)
        ok = (rows < m_lim)[:, None] & (cols < K)[None, :]
        x = tl.load(
            X + rows64[:, None] * stride_x + cols[None, :],
            mask=ok,
            other=0.0,
            cache_modifier=".cg",
        )
        xq, e8 = _compute_mx_quant_and_scale(x, ok, XQ.dtype.element_ty, 0, False)
        tl.store(XQ + rows64[:, None] * stride_q + cols[None, :], xq, mask=ok)
        scols = pid_n * NQ + tl.arange(0, NQ)
        tl.store(
            XS + rows64[:, None] * stride_s + scols[None, :],
            e8,
            mask=(rows < m_lim)[:, None],
        )


@triton.jit
def _ep_mx_quant_tile(
    qid,
    X,
    XQ,
    XS,
    NumLocalTokens,
    M,
    stride_x,
    stride_q,
    stride_s,
    K: tl.constexpr,
    QBM: tl.constexpr,
    QBN: tl.constexpr,
    LIVE_ONLY: tl.constexpr,
    QFP8: tl.constexpr,
):
    """The sort launch's quant CTA body: MXFP8 (QFP8) or MXFP4."""
    if QFP8:
        _ep_mxfp8_quant_tile(
            qid,
            X,
            XQ,
            XS,
            NumLocalTokens,
            M,
            stride_x,
            stride_q,
            stride_s,
            K,
            QBM,
            QBN,
            LIVE_ONLY,
        )
    else:
        _ep_mxfp4_quant_tile(
            qid,
            X,
            XQ,
            XS,
            NumLocalTokens,
            M,
            stride_x,
            stride_q,
            stride_s,
            K,
            QBM,
            QBN,
            LIVE_ONLY,
        )


_ep_topk_repr = make_kernel_repr("_ep_topk_kernel", ["N_EXPTS", "TOPK", "ROWS"])


@triton.jit(repr=_ep_topk_repr)
def _ep_topk_kernel(
    Logits,  # (M, N_EXPTS) bf16 router logits
    DispatchIds,  # (M, TOPK) int32 out
    DispatchWeights,  # (M, TOPK) f32 out
    NumLocalTokens,  # (1,) int32, or None when every row is live
    stride_logits: tl.constexpr,
    n_gates,
    N_EXPTS: tl.constexpr,
    TOPK: tl.constexpr,
    ROWS: tl.constexpr,
):
    """The top-k of ep_sort_routing(logits=...) on its own, ROWS rows a CTA (a wave
    a row), where the sort launches are too few to run it."""
    lim = _ep_gate_limit(NumLocalTokens, n_gates, TOPK)
    _ep_topk_rows(
        Logits,
        stride_logits,
        DispatchIds,
        DispatchWeights,
        tl.program_id(0) * ROWS,
        lim,
        n_gates,
        ROWS,
        TOPK,
        N_EXPTS,
    )


_ep_sort_fused_repr = make_kernel_repr(
    "_ep_sort_fused_kernel",
    [
        "N_EXPTS",
        "TOPK",
        "N_BINS",
        "BLOCK",
        "HIST_BLOCK",
        "HAS_DST_ROW",
        "QUANT",
        "QBM",
        "QFP8",
    ],
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
    # second tile map (HAS_TILES2), last: only the tile writes use them
    TileStart2,  # (N_EXPTS+1,) int32 out, token_offs_pad for tile2_dim_log2
    MDTileInfo2,  # (max_num_tiles2,) int32 out, block_pid_map for tile2_dim_log2
    max_num_tiles2,
    N_EXPTS: tl.constexpr,
    TOPK: tl.constexpr,
    SENTINEL: tl.constexpr,
    N_BINS: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG2_BLOCK: tl.constexpr,
    HIST_BLOCK: tl.constexpr,
    tile_dim_log2: tl.constexpr,
    tile2_dim_log2: tl.constexpr,
    HAS_TILES2: tl.constexpr,
    TILE_BLOCK: tl.constexpr,
    MEMSET_BLOCK: tl.constexpr,
    HAS_DST_ROW: tl.constexpr,
    MAX_TOK: tl.constexpr,
    PEER_ROWS: tl.constexpr,
    ONE_CTA: tl.constexpr,
    X,  # (M, K) bf16 rows to quantize (QUANT only)
    XQ,  # (M, K // 2) uint8 out
    XS,  # (M, K // 32) uint8 out
    n_sort_ctas,
    x_rows,
    stride_x,
    stride_q,
    stride_s,
    QUANT: tl.constexpr,
    QK: tl.constexpr,
    QBM: tl.constexpr,
    QBN: tl.constexpr,
    QUANT_LIVE_ONLY: tl.constexpr,
    QFP8: tl.constexpr,
    Logits,  # (M, N_EXPTS) bf16 router logits (LOGITS only)
    stride_logits: tl.constexpr,
    LOGITS: tl.constexpr,
):
    """EP sort in one launch, for small n_gates (decode).

    Each CTA sorts its own BLOCK gates, but counts ALL gates itself, so it
    knows each expert's first row without a grid-wide join (no atomics, no
    scratch). The redundant counting is cheap while n_gates is small.

    QUANT: the CTAs past the n_sort_ctas sort CTAs MX-quantize X (the sort
    does not read X, so the two halves are independent).

    LOGITS (ONE_CTA, no expert map): the top-k of the router logits runs here
    and writes DispatchIds / DispatchWeights.
    """
    pid = tl.program_id(0)
    n_ctas = n_sort_ctas
    if QUANT:
        qid = pid - n_sort_ctas
        if qid >= 0:
            _ep_mx_quant_tile(
                qid,
                X,
                XQ,
                XS,
                NumLocalTokens,
                x_rows,
                stride_x,
                stride_q,
                stride_s,
                QK,
                QBM,
                QBN,
                QUANT_LIVE_ONLY,
                QFP8,
            )
            return
    start = pid * BLOCK
    lim = _ep_gate_limit(NumLocalTokens, n_gates, TOPK)
    bins = tl.arange(0, N_BINS)
    offs = start + tl.arange(0, BLOCK)
    if LOGITS:
        tl.static_assert(ONE_CTA, "each CTA would need the top-k of all rows")
        _ep_topk_rows(
            Logits,
            stride_logits,
            DispatchIds,
            DispatchWeights,
            0,
            lim,
            n_gates,
            BLOCK // TOPK,
            TOPK,
            N_EXPTS,
        )
        # The gates below read the ids and weights back in gate order.
        tl.debug_barrier()
    valid, expt = _ep_gate_experts(
        DispatchIds, ExpertMap, offs, lim, e_map_numel, SENTINEL
    )
    h = _ep_hist(expt, valid, N_BINS, SENTINEL)
    if ONE_CTA:
        total = h
        ahead = tl.zeros([N_BINS], dtype=tl.int32)
    else:
        total = tl.zeros([N_BINS], dtype=tl.int32)
        ahead = tl.zeros([N_BINS], dtype=tl.int32)
        for c0 in range(0, lim, HIST_BLOCK):
            offs_c = c0 + tl.arange(0, HIST_BLOCK)
            valid_c, expt_c = _ep_gate_experts(
                DispatchIds, ExpertMap, offs_c, lim, e_map_numel, SENTINEL
            )
            h_c = _ep_hist(expt_c, valid_c, N_BINS, SENTINEL)
            total += h_c
            if c0 + HIST_BLOCK <= start:
                ahead += h_c
            elif c0 < start:
                ahead += _ep_hist(expt_c, valid_c & (offs_c < start), N_BINS, SENTINEL)
    n_live = tl.sum(total, 0)

    # ExptData: CTA 0 writes the offsets, CTA c the tiles of experts c + k * n_ctas.
    tl.store(Hist + bins, total, mask=pid == 0)
    n_tiles, tiles_incl, total_tiles = _ep_tile_offsets(
        total, bins, TokenStart, TileStart, pid == 0, N_EXPTS, tile_dim_log2
    )
    if ONE_CTA:
        mine = bins < N_BINS
    else:
        mine = (bins % n_ctas) == pid
    _ep_tile_map(
        n_tiles,
        tiles_incl,
        total_tiles,
        bins,
        mine,
        MDTileInfo,
        max_num_tiles,
        pid,
        n_ctas,
        TILE_BLOCK,
        MEMSET_BLOCK,
    )
    if HAS_TILES2:
        n_tiles2, tiles_incl2, total_tiles2 = _ep_tile_starts(
            total, bins, TileStart2, pid == 0, N_EXPTS, tile2_dim_log2
        )
        _ep_tile_map(
            n_tiles2,
            tiles_incl2,
            total_tiles2,
            bins,
            mine,
            MDTileInfo2,
            max_num_tiles2,
            pid,
            n_ctas,
            TILE_BLOCK,
            MEMSET_BLOCK,
        )

    tl.store(GateValid + offs, valid.to(tl.int32), mask=offs < n_gates)
    expt_base = (tl.cumsum(total, 0) - total) + ahead - (tl.cumsum(h, 0) - h)
    _ep_scatter_sorted(
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
        N_EXPTS,
        BLOCK,
        LOG2_BLOCK,
        HAS_DST_ROW,
        True,
        TOPK,
        MAX_TOK,
        PEER_ROWS,
    )


_ep_gate_hist_scan_repr = make_kernel_repr(
    "_ep_gate_hist_scan_kernel",
    ["N_EXPTS", "TOPK", "N_BINS", "BLOCK", "HAS_DST_ROW", "QUANT", "QBM", "QFP8"],
)


@triton.jit(repr=_ep_gate_hist_scan_repr)
def _ep_gate_hist_scan_kernel(
    DispatchIds,  # (M, topk) int32, GLOBAL expert ids
    ExpertMap,  # (E_map,) int32
    NumLocalTokens,  # (1,) int32, or None when every row is live
    GateValid,  # (G,) int32 out
    CtaHist,  # (n_ctas, N_BINS) int32 out, per-CTA histogram
    HistAtomic,  # (N_BINS,) int32 scratch, ZERO on entry, left ZERO on exit
    Ticket,  # (1,) int32 scratch, ditto -- the grid-wide join counter
    Hist,  # (N_BINS,) int32 out, the durable histogram
    Cursor,  # (N_BINS,) int32 out, first row of each expert
    TokenStart,  # (N_EXPTS+1,) int32 out == token_offs_raw
    TileStart,  # (N_EXPTS+1,) int32 out == token_offs_pad
    MDTileInfo,  # (max_num_tiles,) int32 out == block_pid_map, tail only
    DstRow,  # (G,) int32 out, pre-filled with -1 ("do not deliver")
    max_num_tiles,
    n_gates,
    e_map_numel,
    # second tile map (HAS_TILES2), last: only the JOIN tail uses them
    TileStart2,  # (N_EXPTS+1,) int32 out, token_offs_pad for tile2_dim_log2
    MDTileInfo2,  # (max_num_tiles2,) int32 out, block_pid_map2, tail only
    max_num_tiles2,
    N_EXPTS: tl.constexpr,
    TOPK: tl.constexpr,
    SENTINEL: tl.constexpr,
    N_BINS: tl.constexpr,
    BLOCK: tl.constexpr,
    tile_dim_log2: tl.constexpr,
    tile2_dim_log2: tl.constexpr,
    HAS_TILES2: tl.constexpr,
    MEMSET_BLOCK: tl.constexpr,
    HAS_DST_ROW: tl.constexpr,
    JOIN: tl.constexpr,
    X,  # (M, K) bf16 rows to quantize (QUANT only)
    XQ,  # (M, K // 2) uint8 out
    XS,  # (M, K // 32) uint8 out
    n_sort_ctas,
    x_rows,
    stride_x,
    stride_q,
    stride_s,
    QUANT: tl.constexpr,
    QK: tl.constexpr,
    QBM: tl.constexpr,
    QBN: tl.constexpr,
    QUANT_LIVE_ONLY: tl.constexpr,
    QFP8: tl.constexpr,
    Logits,  # (M, N_EXPTS) bf16 router logits (LOGITS only)
    stride_logits: tl.constexpr,
    DispatchWeights,  # (M, topk) f32 out (LOGITS only)
    LOGITS: tl.constexpr,
):
    """EP sort kernel A: gating + per-CTA histogram (CtaHist).

    QUANT (no JOIN): the CTAs past the n_sort_ctas gate CTAs MX-quantize X.

    LOGITS (no expert map): the top-k of the router logits runs here and writes
    DispatchIds / DispatchWeights for kernel B.

    Without JOIN that is all: each CTA of kernel B sums CtaHist itself.

    With JOIN (very large grids) each CTA also adds its histogram to HistAtomic
    with one atomic per non-empty expert, then takes a ticket. The CTA that draws
    the last ticket owns the scan and ExptData stage1. No CTA waits for another,
    so any occupancy is safe. HistAtomic and Ticket are left zeroed for the next
    call.
    """
    pid = tl.program_id(0)
    if QUANT:
        tl.static_assert(not JOIN, "the join counts every CTA of the grid")
        qid = pid - n_sort_ctas
        if qid >= 0:
            _ep_mx_quant_tile(
                qid,
                X,
                XQ,
                XS,
                NumLocalTokens,
                x_rows,
                stride_x,
                stride_q,
                stride_s,
                QK,
                QBM,
                QBN,
                QUANT_LIVE_ONLY,
                QFP8,
            )
            return
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_gates
    if HAS_DST_ROW and JOIN:
        # Kernel B overwrites the live rows; the tail must stay negative.
        tl.store(DstRow + offs, -1, mask=mask)
    lim = _ep_gate_limit(NumLocalTokens, n_gates, TOPK)
    bins = tl.arange(0, N_BINS)
    if LOGITS:
        # Every row gets its ids and weights, also the dead ones.
        _ep_topk_rows(
            Logits,
            stride_logits,
            DispatchIds,
            DispatchWeights,
            pid * (BLOCK // TOPK),
            lim,
            n_gates,
            BLOCK // TOPK,
            TOPK,
            N_EXPTS,
        )
        # The gates below read the ids back in gate order.
        tl.debug_barrier()
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
    if not JOIN:
        return
    # Finish all waves' histogram atomics before this CTA takes its ticket.
    tl.debug_barrier()

    # Grid-wide join. The last CTA sees every CTA's histogram atomics.
    one = tl.arange(0, 1)
    prev = tl.atomic_add(Ticket + one, 1, sem="acq_rel", scope="gpu")
    if tl.sum(prev, 0) == tl.num_programs(0) - 1:
        h = tl.load(HistAtomic + bins)
        tl.store(Hist + bins, h)
        tl.store(Cursor + bins, tl.cumsum(h, 0) - h)
        # Re-arm the scratch. Safe only here: every other CTA is done with it.
        tl.store(HistAtomic + bins, 0)
        tl.store(Ticket + one, 0)
        _, _, total_tiles = _ep_tile_offsets(
            h, bins, TokenStart, TileStart, True, N_EXPTS, tile_dim_log2
        )
        for t0 in range(total_tiles, max_num_tiles, MEMSET_BLOCK):
            t = t0 + tl.arange(0, MEMSET_BLOCK)
            tl.store(MDTileInfo + t, -1, mask=t < max_num_tiles)
        if HAS_TILES2:
            _, _, total_tiles2 = _ep_tile_starts(
                h, bins, TileStart2, True, N_EXPTS, tile2_dim_log2
            )
            for t0 in range(total_tiles2, max_num_tiles2, MEMSET_BLOCK):
                t = t0 + tl.arange(0, MEMSET_BLOCK)
                tl.store(MDTileInfo2 + t, -1, mask=t < max_num_tiles2)


_ep_sorted_scatter_repr = make_kernel_repr(
    "_ep_sorted_scatter_kernel",
    [
        "N_EXPTS",
        "TOPK",
        "N_BINS",
        "BLOCK",
        "HAS_DST_ROW",
        "JOIN",
        "QUANT",
        "QBM",
        "QFP8",
    ],
)


@triton.jit(repr=_ep_sorted_scatter_repr)
def _ep_sorted_scatter_kernel(
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
    DstRow,  # (G,) int32 out, indexed by sorted position
    SrcTokenMap,  # (max_recv,) int32 in, recv slot -> origin_pe*MAX_TOK + lid
    max_num_tiles,
    n_gates,
    e_map_numel,
    # second tile map (HAS_TILES2), last: only the tile writes use them
    TileStart2,  # (N_EXPTS+1,) int32 in (JOIN) or out, token_offs_pad2
    MDTileInfo2,  # (max_num_tiles2,) int32 out, block_pid_map for tile2_dim_log2
    max_num_tiles2,
    N_EXPTS: tl.constexpr,
    TOPK: tl.constexpr,
    SENTINEL: tl.constexpr,
    N_BINS: tl.constexpr,
    BLOCK: tl.constexpr,
    LOG2_BLOCK: tl.constexpr,
    PREFIX_ROWS: tl.constexpr,
    tile_dim_log2: tl.constexpr,
    tile2_dim_log2: tl.constexpr,
    HAS_TILES2: tl.constexpr,
    MEMSET_BLOCK: tl.constexpr,
    HAS_DST_ROW: tl.constexpr,
    JOIN: tl.constexpr,
    MAX_TOK: tl.constexpr,
    PEER_ROWS: tl.constexpr,
    X,  # (M, K) bf16 rows to quantize (QUANT only)
    XQ,  # (M, K // 2) uint8 out
    XS,  # (M, K // 32) uint8 out
    n_sort_ctas,
    x_rows,
    stride_x,
    stride_q,
    stride_s,
    QUANT: tl.constexpr,
    QK: tl.constexpr,
    QBM: tl.constexpr,
    QBN: tl.constexpr,
    QUANT_LIVE_ONLY: tl.constexpr,
    QFP8: tl.constexpr,
):
    """EP sort kernel B: sorted scatter | ExptData stage2, split by CTA index.

    QUANT: the CTAs past the N_EXPTS + n_sort_ctas sort CTAs MX-quantize X.

    The first N_EXPTS CTAs write MDTileInfo. A scatter CTA ranks its gates by a
    sort; each expert's first row is its offset plus the counts of the CTAs in
    front (rows in ascending gate order). Without JOIN every CTA sums CtaHist
    itself, and also writes the ExptData stage1 outputs of kernel A.
    """
    pid = tl.program_id(0)
    if QUANT:
        qid = pid - N_EXPTS - n_sort_ctas
        if qid >= 0:
            _ep_mx_quant_tile(
                qid,
                X,
                XQ,
                XS,
                NumLocalTokens,
                x_rows,
                stride_x,
                stride_q,
                stride_s,
                QK,
                QBM,
                QBN,
                QUANT_LIVE_ONLY,
                QFP8,
            )
            return
    bins = tl.arange(0, N_BINS)
    if not JOIN:
        # Every CTA sums the whole CtaHist (n_ctas x N_EXPTS) itself; a scatter
        # CTA also sums the rows of the CTAs in front of it.
        n_ctas = n_sort_ctas
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
        if HAS_TILES2:
            tile_start2 = tl.load(TileStart2 + pid)
            _expt_data_compute_stage2(
                pid, Hist, tile_start2, MDTileInfo2, tile2_dim_log2
            )
        tile_start = tl.load(TileStart + pid)
        # Last statement in the branch: stage2 returns early for empty experts.
        _expt_data_compute_stage2(pid, Hist, tile_start, MDTileInfo, tile_dim_log2)
    else:
        # Tiles of expert `pid`.
        is_e = bins == pid
        n_t = tl.sum(tl.where(is_e, n_tiles, 0), 0)
        first_t = tl.sum(tl.where(is_e, tiles_incl - n_tiles, 0), 0)
        for j0 in range(0, n_t, 8):
            j = j0 + tl.arange(0, 8)
            tl.store(MDTileInfo + first_t + j, (j << 16) + pid, mask=j < n_t)
        if HAS_TILES2:
            # The second map is all on the expert CTAs: offsets (CTA 0), the
            # tiles of expert `pid`, and one MEMSET_BLOCK of the tail in turn.
            n_tiles2, tiles_incl2, total_tiles2 = _ep_tile_starts(
                total, bins, TileStart2, pid == 0, N_EXPTS, tile2_dim_log2
            )
            n_t2 = tl.sum(tl.where(is_e, n_tiles2, 0), 0)
            first_t2 = tl.sum(tl.where(is_e, tiles_incl2 - n_tiles2, 0), 0)
            for j0 in range(0, n_t2, 32):
                j = j0 + tl.arange(0, 32)
                tl.store(MDTileInfo2 + first_t2 + j, (j << 16) + pid, mask=j < n_t2)
            for t0 in range(
                total_tiles2 + pid * MEMSET_BLOCK,
                max_num_tiles2,
                N_EXPTS * MEMSET_BLOCK,
            ):
                t = t0 + tl.arange(0, MEMSET_BLOCK)
                tl.store(MDTileInfo2 + t, -1, mask=t < max_num_tiles2)
