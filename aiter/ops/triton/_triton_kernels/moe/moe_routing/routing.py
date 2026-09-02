import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.moe.moe_routing.bitmatrix import (
    _sum_bitmatrix_rows_fused,
)
from aiter.ops.triton._triton_kernels.moe.moe_routing.expt_data import (
    _expt_data_compute_stage1,
    _expt_data_compute_stage2,
    _expt_data_compute_stage2_fused,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit
def _keyed_add(x, y):

    # we keep the key in the upper 16 bits of a uint32:
    key_mask: tl.constexpr = 0xFFFF0000

    kx = x & key_mask
    ky = y & key_mask
    z = tl.where(kx == ky, x + y - kx, y)
    return z


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
            shape=(1, LOAD_SIZE),
            strides=(LOAD_SIZE, 1),
            block_shape=(1, LOAD_SIZE),
        )
        expert = tl.reshape(expt_desc.load([0, 0]), (LOAD_SIZE,))
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
        expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
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
        expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
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
            shape=(1, LOAD_SIZE),
            strides=(LOAD_SIZE, 1),
            block_shape=(1, LOAD_SIZE),
        )
        expert = tl.reshape(expt_desc.load([0, 0]), (LOAD_SIZE,))
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
        expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
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
        expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
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

    _expt_data_compute_stage1(
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
        _expt_data_compute_stage2(pid, ExpertHist, TileStart, MDTileInfo, tile_dim_log2)
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

    _expt_data_compute_stage1(
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
        _expt_data_compute_stage2_fused(pid, ExpertHist, TileStart, MDTileInfo)
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
# The two kernels below serve `ep_sort_routing`, which starts from the output of
# an expert-parallel all-to-all instead of from router logits. The top-k choice
# is already made and the rows have been permuted across ranks, so there is no
# bitmatrix to reduce and only *some* gates of a given row belong to this rank.
# -----------------------------------------------------------------------------


@triton.jit
def _ep_gate_prep_scan_kernel(
    DispatchIds,  # (M, topk) int32, GLOBAL expert ids
    ExpertMap,  # (E_map,) int32
    NumLocalTokens,  # (1,) int32, or None to skip the row mask entirely.
    GateValid,  # (G,) int32 out
    ExptIndx,  # (G,) int32 out, local id or SENTINEL
    HistAtomic,  # (N_BINS,) int32 scratch, ZERO on entry, left ZERO on exit
    Ticket,  # (1,) int32 scratch, ditto -- the grid-wide join counter
    Hist,  # (N_BINS,) int32 out, the durable histogram
    Cursor,  # (N_BINS,) int32 out, bin_base; the scatter bumps it
    TokenStart,  # (N_EXPTS+1,) int32 out == token_offs_raw
    TileStart,  # (N_EXPTS+1,) int32 out == token_offs_pad
    MDTileInfo,  # (max_num_tiles,) int32 out == block_pid_map
    DstRow,  # (G,) int32 out, EP scatter destinations; pre-filled here
    max_num_tiles,
    n_gates,
    e_map_numel,
    N_EXPTS: tl.constexpr,
    TOPK: tl.constexpr,
    SENTINEL: tl.constexpr,
    N_BINS: tl.constexpr,
    BLOCK: tl.constexpr,
    tile_dim_log2: tl.constexpr,
    BLOCK_A: tl.constexpr,
    EQUAL_A: tl.constexpr,
    N_CTAS: tl.constexpr,
    HAS_DST_ROW: tl.constexpr,
):
    """EP sort kernel A: gating + histogram + (in the last CTA) the scan and
    ExptData stage1 -- all in one launch.

    Gating replaces ~9 elementwise launches that all share the G = M*topk axis:
        ids   = dispatch_ids.long().clamp_(0, E_map-1)
        local = expert_map[ids]
        local = where(row < R, local, -1)   # only when NumLocalTokens is given
        gate_valid = (local >= 0).reshape(-1).int()
        expt_indx  = where(local < 0, SENTINEL, local).reshape(-1).int()

    Kernel count is the budget here: on gfx1250 an EMPTY kernel costs ~4.7us of
    device time whatever its grid (measured: grid=1 and grid=2048 are
    indistinguishable), so every launch folded in is 4.7us saved. Two choices
    follow from that:

    1. **No one-hot histogram.** Counting via a (BLOCK, N_BINS) one-hot
       reduction costs N_BINS int ops per gate to count 1 -- 128 at E=48. Here
       the histogram is ``atomic_add`` on LIVE gates only: under EP ~3/4 of
       gates are dead, so the atomic traffic is a quarter of the gate count and
       it spreads over N_EXPTS addresses, nothing like the single-address
       contention that forces an LDS two-level scheme.
    2. **The scan rides along.** The cross-CTA reduction would otherwise force a
       second launch. It is resolved by a ticket instead: each CTA bumps
       ``Ticket`` after its histogram atomics, and whoever draws the last number
       owns the scan. No spin-waiting, so this is safe at any occupancy -- the
       last CTA is by definition the one that had nothing left to wait for.

    ``HistAtomic``/``Ticket`` are caller-persistent and this kernel leaves both
    zeroed, which is what keeps a ``torch.zeros`` launch (another 4.7us) out of
    the steady state.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_gates

    ids = tl.load(DispatchIds + offs, mask=mask, other=0)
    ids = tl.minimum(tl.maximum(ids, 0), e_map_numel - 1)
    local = tl.load(ExpertMap + ids, mask=mask, other=-1)
    valid = (local >= 0) & mask
    if NumLocalTokens is not None:
        row = offs // TOPK
        r = tl.load(NumLocalTokens)
        valid = valid & (row < r)

    tl.store(GateValid + offs, valid.to(tl.int32), mask=mask)
    expt = tl.where(valid, local, SENTINEL).to(tl.int32)
    tl.store(ExptIndx + offs, expt, mask=mask)

    if HAS_DST_ROW:
        # Pre-fill the EP scatter map to "do not deliver". Kernel B overwrites
        # only the sorted positions that a live gate lands on, and the positions
        # it never touches -- the sentinel tail -- must stay negative or the
        # scatter would ship a garbage row. Done here rather than in a memset
        # because this grid already spans exactly [0, n_gates): the *set* of
        # indices covered is the whole map, even though this kernel's axis is the
        # gate index and the map's is the sorted position.
        tl.store(DstRow + offs, -1, mask=mask)

    # `release`: these must be visible to whichever CTA later draws the last
    # ticket. Dead gates are skipped entirely -- the sentinel bin is never
    # counted, so it needs no slot in the prefix sum.
    tl.atomic_add(HistAtomic + expt, 1, mask=valid, sem="release", scope="gpu")

    # Grid-wide join. `acquire` pairs with the release above, so the CTA that
    # sees N_CTAS-1 predecessors also sees all their histogram increments.
    one = tl.arange(0, 1)
    prev = tl.atomic_add(Ticket + one, 1, sem="acq_rel", scope="gpu")
    if tl.sum(prev, 0) == N_CTAS - 1:
        bins = tl.arange(0, N_BINS)
        h = tl.load(HistAtomic + bins)
        tl.store(Hist + bins, h)
        # Exclusive prefix over bins == where each expert's run starts. The
        # scatter takes this as its initial cursor and bumps it per gate.
        tl.store(Cursor + bins, tl.cumsum(h, 0) - h)
        # Re-arm the scratch for the next call. Safe here and only here: drawing
        # the last ticket means every other CTA is done with both buffers.
        tl.store(HistAtomic + bins, 0)
        tl.store(Ticket + one, 0)
        # stage1 re-reads Hist lane-wise under a different offset layout than the
        # store above used, so the CTA has to be made coherent with itself first.
        tl.debug_barrier()
        # pid=0, not `pid`: stage1 is pid-independent apart from its terminal
        # writes and the 0xFFFFFFFF tail memset, which are exactly what the
        # `pid == 0` guard inside it selects. One CTA is enough -- letting all
        # N_EXPTS of them recompute the identical prefix sums buys nothing.
        _expt_data_compute_stage1(
            0,
            Hist,
            N_EXPTS,
            TokenStart,
            TileStart,
            MDTileInfo,
            max_num_tiles,
            n_gates,
            tile_dim_log2,
            BLOCK_A,
            EQUAL_A,
        )


@triton.jit
def _ep_scatter_atomic_expt_data_kernel(
    ExptIndx,  # (G,) int32 in
    DispatchWeights,  # (M, topk) f32 in, read flat
    Cursor,  # (N_BINS,) int32 in/out, bin_base on entry
    GatherIndx,  # (G,) int32 out
    ScatterIndx,  # (G,) int32 out
    GateScal,  # (G,) f32 out
    Hist,  # (N_BINS,) int32 in -- stage2 half only
    TileStart,  # (N_EXPTS+1,) int32 in -- stage2 half only
    MDTileInfo,  # (max_num_tiles,) int32 out -- stage2 half only
    # --- EP scatter map, scatter half only (see ep_sort_routing) ---
    DstRow,  # (G,) int32 out, indexed by sorted position
    SrcTokenMap,  # (max_recv,) int32 in, recv slot -> origin_pe*MAX_TOK + lid
    n_gates,
    N_EXPTS: tl.constexpr,
    SENTINEL: tl.constexpr,
    tile_dim_log2: tl.constexpr,
    GATE_BLOCK: tl.constexpr,
    HAS_DST_ROW: tl.constexpr,
    TOPK: tl.constexpr,
    MAX_TOK: tl.constexpr,
    PEER_ROWS: tl.constexpr,
):
    """EP sort kernel B: atomic-cursor scatter | ExptData stage2, split by CTA
    index.

    Grid is ``(N_EXPTS + n_ctas,)`` with an ``if pid >= N_EXPTS`` split, the same
    layout ``_combined_routing_fused`` uses. Legal because, given ``Hist``, the
    two halves are independent: both only *read* the histogram, and their writes
    are disjoint -- the scatter owns GatherIndx/ScatterIndx/GateScal, stage2 owns
    MDTileInfo. No barrier is needed before the split either, since Hist was
    finalised in kernel A, one launch boundary back.

    A gate's destination is ``atomic_add(&Cursor[e], 1)``, since Cursor starts at
    the expert's base. The alternative -- deriving rank from a ``tl.cumsum`` over
    a (GATE_BLOCK, N_BINS) one-hot, 32K int32 of live values per CTA -- measured
    29.10us against 7.96us for this version on the same input.

    ORDER WITHIN AN EXPERT IS NOT ASCENDING BY GATE INDEX. That is invisible
    downstream and the final output stays bitwise reproducible: each sorted row's
    dot product is independent of where it lands, and the combine sums per
    (token, slot) via ScatterIndx, not in sorted order.

    Cost of the branch fusion: one register/LDS budget and one ``num_warps`` for
    both halves, sized by whichever is heavier.
    """
    pid = tl.program_id(0)
    if pid >= N_EXPTS:
        idx = (pid - N_EXPTS) * GATE_BLOCK + tl.arange(0, GATE_BLOCK)
        mask = idx < n_gates
        expt = tl.load(ExptIndx + idx, mask=mask, other=SENTINEL)
        live = mask & (expt != SENTINEL)
        # Contention is per-expert, not global: ~hist[e]/n_ctas lanes per address.
        pos = tl.atomic_add(Cursor + expt, 1, mask=live, sem="relaxed", scope="gpu")
        tl.store(GatherIndx + pos, idx.to(tl.int32), mask=live)
        # Dead gates get 0 -- reduce_grouped clamps them via indx_valid before
        # dereferencing. Written unmasked over `mask` so the buffer needs no
        # separate memset.
        tl.store(ScatterIndx + idx, tl.where(live, pos, 0).to(tl.int32), mask=mask)
        w = tl.load(DispatchWeights + idx, mask=live, other=0.0)
        tl.store(GateScal + pos, w.to(tl.float32), mask=live)
        if HAS_DST_ROW:
            # Where GEMM2's row for this gate has to land in the combine staging
            # window: the origin rank's slot for (its token, k). `pos` is the row
            # the GEMM will produce, so the map is keyed by it and the scatter
            # needs no second indirection.
            #
            # One row index selects both peer and slot, because every peer's slot
            # region sits at the same stride in the symmetric window. Free here:
            # `pos`, `idx` and `live` are already in registers.
            recv_slot = idx // TOPK
            k = idx - recv_slot * TOPK
            enc = tl.load(SrcTokenMap + recv_slot, mask=live, other=0)
            origin_pe = enc // MAX_TOK
            origin_lid = enc - origin_pe * MAX_TOK
            dst = origin_pe * PEER_ROWS + origin_lid * TOPK + k
            tl.store(DstRow + pos, dst.to(tl.int32), mask=live)
    else:
        # Last statement in the branch on purpose: stage2 early-returns for empty
        # experts, so nothing may follow it.
        _expt_data_compute_stage2(pid, Hist, TileStart, MDTileInfo, tile_dim_log2)
